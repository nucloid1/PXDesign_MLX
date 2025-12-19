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
MLX implementation of quaternion affine transformations for AlphaFold2.

Based on DeepMind's quat_affine.py implementation.
"""

import mlx.core as mx
import numpy as np
from typing import List, Tuple


# Quaternion to rotation matrix conversion constants
QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)
QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]  # kk
QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]  # jk
QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # kr


def quat_to_rot(normalized_quat: mx.array) -> mx.array:
    """
    Convert normalized quaternion to rotation matrix.

    Args:
        normalized_quat: [..., 4] quaternion (normalized)

    Returns:
        Rotation matrix [..., 3, 3]
    """
    # Convert to MLX array
    quat_to_rot_tensor = mx.array(QUAT_TO_ROT)

    # Compute rotation matrix using einsum
    # rot_tensor = sum over i,j of QUAT_TO_ROT[i,j] * quat[i] * quat[j]
    rot_tensor = mx.sum(
        mx.reshape(quat_to_rot_tensor, (4, 4, 9)) *
        normalized_quat[..., :, None, None] *
        normalized_quat[..., None, :, None],
        axis=(-3, -2)
    )

    # Reshape to [..., 3, 3]
    return mx.reshape(rot_tensor, (*normalized_quat.shape[:-1], 3, 3))


def quat_multiply(quat1: mx.array, quat2: mx.array) -> mx.array:
    """
    Multiply two quaternions.

    Args:
        quat1: [..., 4] first quaternion
        quat2: [..., 4] second quaternion

    Returns:
        Product quaternion [..., 4]
    """
    # Hamilton product
    w1, x1, y1, z1 = mx.split(quat1, 4, axis=-1)
    w2, x2, y2, z2 = mx.split(quat2, 4, axis=-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return mx.concatenate([w, x, y, z], axis=-1)


def apply_rot_to_vec(rot: mx.array, vec: mx.array) -> mx.array:
    """
    Apply rotation matrix to vector.

    Args:
        rot: [..., 3, 3] rotation matrix
        vec: [..., 3] or [..., D, 3] vector (D = extra dimensions)

    Returns:
        Rotated vector [..., 3] or [..., D, 3]
    """
    # Check if vec has extra dimensions
    if vec.ndim > rot.ndim - 1:
        # vec: [..., D, 3], rot: [..., 3, 3]
        # Need to expand rot to [..., 1, 3, 3] for broadcasting
        num_extra_dims = vec.ndim - rot.ndim + 1
        for _ in range(num_extra_dims):
            rot = mx.expand_dims(rot, axis=-3)
        # Now rot: [..., 1, 3, 3], vec: [..., D, 3]
        # Expand vec: [..., D, 1, 3]
        vec_expanded = mx.expand_dims(vec, axis=-2)
        # Multiply and sum
        return mx.sum(rot * vec_expanded, axis=-1)  # [..., D, 3]
    else:
        # Standard case: vec: [..., 3], rot: [..., 3, 3]
        return mx.sum(rot * mx.expand_dims(vec, axis=-2), axis=-1)


def invert_rot(rot: mx.array) -> mx.array:
    """
    Invert rotation matrix (transpose for orthogonal matrices).

    Args:
        rot: [..., 3, 3] rotation matrix

    Returns:
        Inverted rotation [..., 3, 3]
    """
    return mx.transpose(rot, axes=list(range(len(rot.shape)-2)) + [-1, -2])


class QuatAffine:
    """
    Quaternion affine transformation (rotation + translation).

    Represents coordinate frames with:
    - quaternion: [..., 4] unit quaternion representing rotation
    - translation: [..., 3] translation vector
    - rotation: [..., 3, 3] rotation matrix (derived from quaternion)
    """

    def __init__(
        self,
        quaternion: mx.array,
        translation: mx.array,
        rotation: mx.array = None,
        normalize: bool = True
    ):
        """
        Initialize QuatAffine.

        Args:
            quaternion: [..., 4] quaternion
            translation: [..., 3] translation
            rotation: [..., 3, 3] rotation matrix (optional, computed from quat if not provided)
            normalize: Whether to normalize the quaternion
        """
        if normalize:
            quaternion = quaternion / (mx.sqrt(mx.sum(mx.square(quaternion), axis=-1, keepdims=True)) + 1e-8)

        self.quaternion = quaternion
        self.translation = translation

        if rotation is None:
            self.rotation = quat_to_rot(quaternion)
        else:
            self.rotation = rotation

    def apply_to_point(self, point: mx.array) -> mx.array:
        """
        Apply transformation to points.

        Args:
            point: [..., 3] or list of 3 arrays representing x, y, z coordinates

        Returns:
            Transformed point [..., 3] or list of 3 arrays
        """
        # Handle list input (from IPA)
        if isinstance(point, list):
            # point is list of 3 arrays, each with shape [..., D]
            # Stack to [..., D, 3]
            point_array = mx.stack(point, axis=-1)
            # Apply rotation and translation
            result = apply_rot_to_vec(self.rotation, point_array)
            # Add translation - need to broadcast properly
            # result: [..., D, 3], self.translation: [..., 3]
            # Expand translation: [..., 1, 3]
            trans_expanded = mx.expand_dims(self.translation, axis=-2)
            result = result + trans_expanded
            # Return as list of 3 arrays, each [..., D]
            return [result[..., i] for i in range(3)]
        else:
            return apply_rot_to_vec(self.rotation, point) + self.translation

    def invert_point(self, point: mx.array) -> mx.array:
        """
        Apply inverse transformation to points (global to local frame).

        Args:
            point: [..., 3] or list of 3 arrays

        Returns:
            Transformed point [..., 3] or list of 3 arrays
        """
        # Handle list input
        if isinstance(point, list):
            # point is list of 3 arrays, each with shape [..., D]
            # Stack to [..., D, 3]
            point_array = mx.stack(point, axis=-1)
            inv_rot = invert_rot(self.rotation)
            # Subtract translation - need to broadcast
            # point_array: [..., D, 3], self.translation: [..., 3]
            trans_expanded = mx.expand_dims(self.translation, axis=-2)
            result = apply_rot_to_vec(inv_rot, point_array - trans_expanded)
            # Return as list of 3 arrays, each [..., D]
            return [result[..., i] for i in range(3)]
        else:
            inv_rot = invert_rot(self.rotation)
            return apply_rot_to_vec(inv_rot, point - self.translation)

    def compose(self, other: 'QuatAffine') -> 'QuatAffine':
        """
        Compose two transformations.

        Args:
            other: Another QuatAffine

        Returns:
            Composed transformation
        """
        new_quat = quat_multiply(self.quaternion, other.quaternion)
        new_trans = self.apply_to_point(other.translation)
        return QuatAffine(new_quat, new_trans, normalize=True)

    @staticmethod
    def identity(shape: Tuple, dtype=mx.float32) -> 'QuatAffine':
        """
        Create identity transformation.

        Args:
            shape: Shape of batch dimensions
            dtype: Data type

        Returns:
            Identity QuatAffine
        """
        quat = mx.zeros((*shape, 4), dtype=dtype)
        quat = quat.at[..., 0].add(1.0)  # [1, 0, 0, 0] = identity rotation
        trans = mx.zeros((*shape, 3), dtype=dtype)
        return QuatAffine(quat, trans, normalize=False)

    @staticmethod
    def from_tensor(tensor: mx.array) -> 'QuatAffine':
        """
        Create QuatAffine from tensor.

        Args:
            tensor: [..., 7] tensor with [quaternion (4), translation (3)]

        Returns:
            QuatAffine object
        """
        quaternion = tensor[..., :4]
        translation = tensor[..., 4:]
        return QuatAffine(quaternion, translation, normalize=True)

    def to_tensor(self) -> mx.array:
        """
        Convert to tensor representation.

        Returns:
            [..., 7] tensor with [quaternion (4), translation (3)]
        """
        return mx.concatenate([self.quaternion, self.translation], axis=-1)


def squared_difference(x: mx.array, y: mx.array) -> mx.array:
    """Compute squared difference between tensors."""
    return mx.square(x - y)
