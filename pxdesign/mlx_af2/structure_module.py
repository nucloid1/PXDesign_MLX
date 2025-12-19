# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MLX implementation of Structure Module for AlphaFold2.

Based on Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional

from .quat_affine import QuatAffine, squared_difference
from .evoformer import LayerNorm


class InvariantPointAttention(nn.Module):
    """
    Invariant Point Attention module.

    Geometry-aware attention that operates on points in 3D space.
    Attention is based on Euclidean distances between query and key points
    in the global frame.

    Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"

    Args:
        c_s: Single (node) representation channels
        c_z: Pair representation channels
        num_heads: Number of attention heads
        num_qk_points: Number of query/key points
        num_v_points: Number of value points
        num_scalar_qk: Number of scalar query/key features
        num_scalar_v: Number of scalar value features
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        num_heads: int = 12,
        num_qk_points: int = 4,
        num_v_points: int = 8,
        num_scalar_qk: int = 16,
        num_scalar_v: int = 16,
        dist_epsilon: float = 1e-8
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.dist_epsilon = dist_epsilon

        # Scalar queries
        self.q_scalar = nn.Linear(c_s, num_heads * num_scalar_qk, bias=False)

        # Scalar keys and values
        self.kv_scalar = nn.Linear(c_s, num_heads * (num_scalar_qk + num_scalar_v), bias=False)

        # Query points (in local frame)
        self.q_point_local = nn.Linear(c_s, num_heads * 3 * num_qk_points, bias=False)

        # Key and value points (in local frame)
        self.kv_point_local = nn.Linear(c_s, num_heads * 3 * (num_qk_points + num_v_points), bias=False)

        # Attention bias from pair representation
        self.attention_2d = nn.Linear(c_z, num_heads, bias=False)

        # Output projection
        output_dim = (
            num_heads * num_scalar_v +  # scalar values
            num_heads * num_v_points * 3 +  # point values (x, y, z)
            num_heads * num_v_points +  # point norms
            num_heads * c_z  # attention over pair features
        )
        self.output_projection = nn.Linear(output_dim, c_s, bias=True)

        # Trainable point weights
        self.trainable_point_weights = mx.ones((num_heads,)) * np.log(np.exp(1.0) - 1.0)

        # Variance scaling factors
        scalar_variance = max(num_scalar_qk, 1) * 1.0
        point_variance = max(num_qk_points, 1) * 9.0 / 2.0
        num_logit_terms = 3
        self.scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance))
        self.point_weights_base = np.sqrt(1.0 / (num_logit_terms * point_variance))
        self.attention_2d_weights = np.sqrt(1.0 / num_logit_terms)

    def __call__(
        self,
        inputs_1d: mx.array,
        inputs_2d: mx.array,
        mask: mx.array,
        affine: QuatAffine
    ) -> mx.array:
        """
        Forward pass.

        Args:
            inputs_1d: [N_res, c_s] single representation
            inputs_2d: [N_res, N_res, c_z] pair representation
            mask: [N_res, 1] mask
            affine: QuatAffine representing backbone frames

        Returns:
            Updated single representation [N_res, c_s]
        """
        N_res = inputs_1d.shape[0]

        # Scalar queries: [N_res, num_heads, num_scalar_qk]
        q_scalar = self.q_scalar(inputs_1d)
        q_scalar = mx.reshape(q_scalar, (N_res, self.num_heads, self.num_scalar_qk))

        # Scalar keys and values: [N_res, num_heads, num_scalar_qk + num_scalar_v]
        kv_scalar = self.kv_scalar(inputs_1d)
        kv_scalar = mx.reshape(kv_scalar, (N_res, self.num_heads, self.num_scalar_qk + self.num_scalar_v))
        k_scalar = kv_scalar[..., :self.num_scalar_qk]
        v_scalar = kv_scalar[..., self.num_scalar_qk:]

        # Query points in local frame
        q_point_local = self.q_point_local(inputs_1d)
        # Split into x, y, z components
        q_point_local = mx.split(q_point_local, 3, axis=-1)
        # Transform to global frame: list of 3 arrays [N_res, num_heads * num_qk_points]
        q_point_global = affine.apply_to_point(q_point_local)
        # Reshape each component: [N_res, num_heads, num_qk_points]
        q_point = [mx.reshape(x, (N_res, self.num_heads, self.num_qk_points)) for x in q_point_global]

        # Key and value points in local frame
        kv_point_local = self.kv_point_local(inputs_1d)
        kv_point_local = mx.split(kv_point_local, 3, axis=-1)
        # Transform to global frame
        kv_point_global = affine.apply_to_point(kv_point_local)
        # Reshape: [N_res, num_heads, num_qk_points + num_v_points]
        kv_point_global = [
            mx.reshape(x, (N_res, self.num_heads, self.num_qk_points + self.num_v_points))
            for x in kv_point_global
        ]
        # Split into keys and values
        k_point = [x[..., :self.num_qk_points] for x in kv_point_global]
        v_point = [x[..., self.num_qk_points:] for x in kv_point_global]

        # Compute attention logits

        # 1. Scalar attention: [num_heads, N_res, N_res]
        q_scalar = mx.transpose(q_scalar * self.scalar_weights, (1, 0, 2))  # [num_heads, N_res, num_scalar_qk]
        k_scalar = mx.transpose(k_scalar, (1, 0, 2))  # [num_heads, N_res, num_scalar_qk]
        attn_qk_scalar = mx.matmul(q_scalar, mx.transpose(k_scalar, (0, 2, 1)))  # [num_heads, N_res, N_res]

        # 2. Point attention: [num_heads, N_res, N_res]
        # Compute squared distances between query and key points
        # Swap axes: [num_heads, N_res, num_qk_points]
        q_point = [mx.transpose(x, (1, 0, 2)) for x in q_point]
        k_point = [mx.transpose(x, (1, 0, 2)) for x in k_point]

        # Compute pairwise squared distances for each point
        dist2 = []
        for qx, kx in zip(q_point, k_point):
            # qx: [num_heads, N_res, num_qk_points]
            # Expand: [num_heads, N_res, 1, num_qk_points] and [num_heads, 1, N_res, num_qk_points]
            d2 = squared_difference(
                mx.expand_dims(qx, axis=2),
                mx.expand_dims(kx, axis=1)
            )  # [num_heads, N_res, N_res, num_qk_points]
            dist2.append(d2)

        # Sum over x, y, z: [num_heads, N_res, N_res, num_qk_points]
        dist2 = sum(dist2)

        # Apply trainable point weights
        point_weights = self.point_weights_base * nn.softplus(self.trainable_point_weights)
        point_weights = mx.reshape(mx.array(point_weights), [self.num_heads, 1, 1, 1])

        # Sum over points: [num_heads, N_res, N_res]
        attn_qk_point = -0.5 * mx.sum(point_weights * dist2, axis=-1)

        # 3. Pair representation bias: [num_heads, N_res, N_res]
        attention_2d = self.attention_2d(inputs_2d)  # [N_res, N_res, num_heads]
        attention_2d = mx.transpose(attention_2d, (2, 0, 1))  # [num_heads, N_res, N_res]
        attention_2d = attention_2d * self.attention_2d_weights

        # Combine all attention components
        attn_logits = attn_qk_scalar + attn_qk_point + attention_2d

        # Apply mask
        mask_2d = mask * mx.transpose(mask, (1, 0))  # [N_res, N_res]
        attn_logits = attn_logits - 1e5 * (1.0 - mask_2d)

        # Softmax: [num_heads, N_res, N_res]
        attn = mx.softmax(attn_logits, axis=-1)

        # Apply attention to values

        # 1. Scalar values: [num_heads, N_res, num_scalar_v]
        v_scalar = mx.transpose(v_scalar, (1, 0, 2))  # [num_heads, N_res, num_scalar_v]
        result_scalar = mx.matmul(attn, v_scalar)  # [num_heads, N_res, num_scalar_v]
        result_scalar = mx.transpose(result_scalar, (1, 0, 2))  # [N_res, num_heads, num_scalar_v]
        result_scalar = mx.reshape(result_scalar, (N_res, self.num_heads * self.num_scalar_v))

        # 2. Point values: [num_heads, N_res, num_v_points] for each of x, y, z
        v_point = [mx.transpose(x, (1, 0, 2)) for x in v_point]  # [num_heads, N_res, num_v_points]
        result_point_global = []
        for vx in v_point:
            # attn: [num_heads, N_res, N_res]
            # vx: [num_heads, N_res, num_v_points]
            # Manual einsum: bhqk,bhkc->bhqc
            result = mx.sum(
                mx.expand_dims(attn, axis=-1) * mx.expand_dims(vx, axis=1),
                axis=2
            )  # [num_heads, N_res, num_v_points]
            result_point_global.append(result)

        # Transpose back: [N_res, num_heads, num_v_points]
        result_point_global = [mx.transpose(x, (1, 0, 2)) for x in result_point_global]

        # Reshape for output: [N_res, num_heads * num_v_points]
        result_point_global = [mx.reshape(x, (N_res, self.num_heads * self.num_v_points)) for x in result_point_global]

        # Transform to local frame
        result_point_local = affine.invert_point(result_point_global)

        # Compute point norms
        point_norm = mx.sqrt(
            self.dist_epsilon +
            mx.square(result_point_local[0]) +
            mx.square(result_point_local[1]) +
            mx.square(result_point_local[2])
        )  # [N_res, num_heads * num_v_points]

        # 3. Attention over pair features
        # attn: [num_heads, N_res, N_res]
        # inputs_2d: [N_res, N_res, c_z]
        result_attention_over_2d = mx.einsum('hij,ijc->ihc', attn, inputs_2d)  # [num_heads, N_res, c_z]
        result_attention_over_2d = mx.reshape(
            result_attention_over_2d,
            (N_res, self.num_heads * self.c_z)
        )

        # Concatenate all output features
        output_features = [
            result_scalar,
            result_point_local[0],
            result_point_local[1],
            result_point_local[2],
            point_norm,
            result_attention_over_2d
        ]
        final_act = mx.concatenate(output_features, axis=-1)

        # Output projection
        return self.output_projection(final_act)


class BackboneUpdate(nn.Module):
    """
    Update backbone frames using predicted rotations and translations.

    Args:
        c_s: Single representation channels
    """

    def __init__(self, c_s: int = 384):
        super().__init__()
        self.linear = nn.Linear(c_s, 6, bias=True)  # 6 = quaternion update (3) + translation update (3)

    def __call__(self, s: mx.array, affine: QuatAffine) -> QuatAffine:
        """
        Update backbone frames.

        Args:
            s: Single representation [N_res, c_s]
            affine: Current QuatAffine

        Returns:
            Updated QuatAffine
        """
        # Predict update
        update = self.linear(s)  # [N_res, 6]

        # Split into rotation and translation updates
        quat_update_vec = update[..., :3]  # [N_res, 3]
        trans_update = update[..., 3:]  # [N_res, 3]

        # Convert rotation update to quaternion
        # Create quaternion from axis-angle representation
        # quat = [cos(theta/2), sin(theta/2) * axis]
        angle = mx.sqrt(mx.sum(mx.square(quat_update_vec), axis=-1, keepdims=True) + 1e-8)
        axis = quat_update_vec / (angle + 1e-8)

        # Build quaternion
        quat_w = mx.cos(angle)
        quat_xyz = mx.sin(angle) * axis
        quat_update = mx.concatenate([quat_w, quat_xyz], axis=-1)  # [N_res, 4]

        # Apply update to current affine
        # New quaternion = current_quat * update_quat
        from .quat_affine import quat_multiply
        new_quat = quat_multiply(affine.quaternion, quat_update)

        # Apply translation update in current frame
        new_trans = affine.translation + affine.apply_to_point(trans_update)

        return QuatAffine(new_quat, new_trans, normalize=True)


class StructureModuleIteration(nn.Module):
    """
    Single iteration of structure module.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" lines 6-21

    Args:
        c_s: Single representation channels
        c_z: Pair representation channels
        ... (IPA args)
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        **ipa_kwargs
    ):
        super().__init__()

        self.layer_norm_s = LayerNorm(c_s, affine=True)
        self.layer_norm_z = LayerNorm(c_z, affine=True)

        self.ipa = InvariantPointAttention(c_s=c_s, c_z=c_z, **ipa_kwargs)

        # Transition after IPA
        self.transition = nn.Sequential(
            LayerNorm(c_s, affine=True),
            nn.Linear(c_s, c_s, bias=True),
            nn.ReLU(),
            nn.Linear(c_s, c_s, bias=True),
            nn.ReLU(),
            nn.Linear(c_s, c_s, bias=True)
        )

        self.backbone_update = BackboneUpdate(c_s=c_s)

    def __call__(
        self,
        s: mx.array,
        z: mx.array,
        affine: QuatAffine,
        mask: mx.array
    ) -> tuple[mx.array, QuatAffine]:
        """
        Forward pass.

        Args:
            s: Single representation [N_res, c_s]
            z: Pair representation [N_res, N_res, c_z]
            affine: Current backbone frames
            mask: Residue mask [N_res, 1]

        Returns:
            Updated (s, affine)
        """
        # IPA with residual
        s = s + self.ipa(self.layer_norm_s(s), self.layer_norm_z(z), mask, affine)

        # Transition with residual
        s = s + self.transition(s)

        # Update backbone
        affine = self.backbone_update(s, affine)

        return s, affine


class StructureModule(nn.Module):
    """
    Complete structure module for predicting 3D coordinates.

    Args:
        num_iterations: Number of structure module iterations (typically 8)
        c_s: Single representation channels
        c_z: Pair representation channels
    """

    def __init__(
        self,
        num_iterations: int = 8,
        c_s: int = 384,
        c_z: int = 128,
        **ipa_kwargs
    ):
        super().__init__()
        self.num_iterations = num_iterations

        self.iterations = [
            StructureModuleIteration(c_s=c_s, c_z=c_z, **ipa_kwargs)
            for _ in range(num_iterations)
        ]

    def __call__(
        self,
        s: mx.array,
        z: mx.array,
        mask: mx.array,
        initial_affine: Optional[QuatAffine] = None
    ) -> tuple[mx.array, QuatAffine]:
        """
        Forward pass.

        Args:
            s: Single representation [N_res, c_s]
            z: Pair representation [N_res, N_res, c_z]
            mask: Residue mask [N_res, 1]
            initial_affine: Initial backbone frames (optional)

        Returns:
            Final (s, affine)
        """
        N_res = s.shape[0]

        # Initialize affine if not provided
        if initial_affine is None:
            affine = QuatAffine.identity((N_res,), dtype=s.dtype)
        else:
            affine = initial_affine

        # Run iterations
        for iteration in self.iterations:
            s, affine = iteration(s, z, affine, mask)

        return s, affine
