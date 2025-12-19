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
MLX implementation of Evoformer block for AlphaFold2.

Based on Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack"
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict

from .attention import MLXAttention


class LayerNorm(nn.Module):
    """Layer normalization with optional affine transformation."""

    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.dims = dims
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        if self.weight is not None:
            x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias

        return x_norm


class Transition(nn.Module):
    """
    Transition layer (MLP) for MSA and pair representations.

    Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
    Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"

    Args:
        c_in: Input channels
        num_intermediate_factor: Expansion factor for hidden layer (typically 4)
    """

    def __init__(self, c_in: int, num_intermediate_factor: int = 4):
        super().__init__()
        c_hidden = c_in * num_intermediate_factor

        self.layer_norm = LayerNorm(c_in, affine=True)
        self.linear1 = nn.Linear(c_in, c_hidden)
        self.linear2 = nn.Linear(c_hidden, c_in)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor [..., c_in]
            mask: Optional mask [..., 1]

        Returns:
            Output tensor [..., c_in]
        """
        # Layer norm
        x = self.layer_norm(x)

        # MLP
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)

        # Apply mask if provided
        if mask is not None:
            x = x * mask

        return x


class TriangleMultiplication(nn.Module):
    """
    Triangle multiplication layer (outgoing or incoming edges).

    Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
    Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"

    Args:
        c_z: Pair representation channels
        c_hidden: Hidden channels
        equation: Einsum equation ('ikc,jkc->ijc' for outgoing, 'kjc,kic->ijc' for incoming)
    """

    def __init__(self, c_z: int, c_hidden: int = 128, equation: str = 'ikc,jkc->ijc'):
        super().__init__()
        self.equation = equation
        self.c_hidden = c_hidden

        self.layer_norm_input = LayerNorm(c_z, affine=True)
        self.left_projection = nn.Linear(c_z, c_hidden, bias=False)
        self.right_projection = nn.Linear(c_z, c_hidden, bias=False)
        self.left_gate = nn.Linear(c_z, c_hidden, bias=True)
        self.right_gate = nn.Linear(c_z, c_hidden, bias=True)

        self.center_layer_norm = LayerNorm(c_hidden, affine=True)
        self.output_projection = nn.Linear(c_hidden, c_z, bias=True)
        self.gating_linear = nn.Linear(c_z, c_z, bias=True)

    def __call__(self, pair_act: mx.array, pair_mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass.

        Args:
            pair_act: Pair activations [N_res, N_res, c_z]
            pair_mask: Pair mask [N_res, N_res]

        Returns:
            Updated pair activations [N_res, N_res, c_z]
        """
        # Expand mask to [..., 1]
        if pair_mask is not None:
            mask = mx.expand_dims(pair_mask, axis=-1)
        else:
            mask = 1.0

        # Layer norm
        act = self.layer_norm_input(pair_act)
        input_act = act

        # Projections
        left_proj = mask * self.left_projection(act)
        right_proj = mask * self.right_projection(act)

        # Gating
        left_gate = mx.sigmoid(self.left_gate(act))
        right_gate = mx.sigmoid(self.right_gate(act))

        left_proj = left_proj * left_gate
        right_proj = right_proj * right_gate

        # Triangle multiplication via einsum
        # For outgoing: 'ikc,jkc->ijc' (left_proj[i,k,c] * right_proj[j,k,c] -> out[i,j,c])
        # For incoming: 'kjc,kic->ijc' (left_proj[k,j,c] * right_proj[k,i,c] -> out[i,j,c])
        if self.equation == 'ikc,jkc->ijc':
            # Outgoing: sum over k
            act = mx.einsum('ikc,jkc->ijc', left_proj, right_proj)
        else:
            # Incoming: sum over k
            act = mx.einsum('kjc,kic->ijc', left_proj, right_proj)

        # Center layer norm
        act = self.center_layer_norm(act)

        # Output projection
        act = self.output_projection(act)

        # Output gating
        gate = mx.sigmoid(self.gating_linear(input_act))
        act = act * gate

        return act


class TriangleAttention(nn.Module):
    """
    Triangle Attention (starting or ending node).

    Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
    Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"

    Args:
        c_z: Pair representation channels
        c_hidden: Hidden dimension per head
        num_heads: Number of attention heads
        orientation: 'per_row' for starting node, 'per_column' for ending node
    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int = 32,
        num_heads: int = 4,
        orientation: str = 'per_row'
    ):
        super().__init__()
        self.orientation = orientation
        self.num_heads = num_heads

        self.layer_norm = LayerNorm(c_z, affine=True)
        self.attention = MLXAttention(
            c_q=c_z,
            c_kv=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            c_out=c_z,
            gating=True
        )

        # Linear layer to create bias from pair features
        self.feat_2d_proj = nn.Linear(c_z, num_heads, bias=False)

    def __call__(self, pair_act: mx.array, pair_mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass.

        Args:
            pair_act: Pair activations [N_res, N_res, c_z]
            pair_mask: Pair mask [N_res, N_res]

        Returns:
            Updated pair activations [N_res, N_res, c_z]
        """
        # Swap axes if per_column orientation
        if self.orientation == 'per_column':
            pair_act = mx.transpose(pair_act, (1, 0, 2))
            if pair_mask is not None:
                pair_mask = mx.transpose(pair_mask, (1, 0))

        # Layer norm
        pair_act = self.layer_norm(pair_act)

        # Create bias from pair features (non-batched bias, same for all rows)
        # [N_res, N_res, c_z] -> [N_res, N_res, num_heads]
        bias = self.feat_2d_proj(pair_act)
        # Transpose to [num_heads, N_res, N_res] - this is the same bias for all batch elements
        bias = mx.transpose(bias, (2, 0, 1))
        # Expand to [N_res, num_heads, N_res, N_res] by broadcasting
        # Each of the N_res batch elements uses the same bias pattern
        N_res = pair_act.shape[0]
        bias = mx.broadcast_to(
            mx.expand_dims(bias, axis=0),  # [1, num_heads, N_res, N_res]
            (N_res, self.num_heads, N_res, N_res)  # [N_res, num_heads, N_res, N_res]
        )

        # Add mask to bias if provided
        if pair_mask is not None:
            # [N_res, N_res] -> [N_res, 1, 1, N_res] for broadcasting with attention bias
            mask_bias = (1e9 * (pair_mask - 1.0))
            # For each batch element (row i), mask is pair_mask[i, :]
            # Expand to [N_res, 1, 1, N_res] (batch, heads, query, key)
            mask_bias = mx.expand_dims(mx.expand_dims(mask_bias, axis=1), axis=1)  # [N_res, 1, 1, N_res]
            # Add mask bias to attention bias
            bias = bias + mask_bias

        # For per_row attention, treat each row as a batch element
        # pair_act is [N_res, N_res, c_z] which is already [batch, positions, channels]
        # Apply attention
        output = self.attention(
            pair_act,  # [N_res, N_res, c_z] as [batch, N_q, c_q]
            pair_act,  # [N_res, N_res, c_z] as [batch, N_kv, c_kv]
            bias=bias,  # [N_res, num_heads, N_res, N_res]
            mask=None  # Already included in bias
        )

        # Swap axes back if per_column orientation
        if self.orientation == 'per_column':
            output = mx.transpose(output, (1, 0, 2))

        return output


class MSARowAttentionWithPairBias(nn.Module):
    """
    MSA per-row attention biased by the pair representation.

    Jumper et al. (2021) Suppl. Alg. 7 "MSARowAttentionWithPairBias"

    Args:
        c_m: MSA representation channels
        c_z: Pair representation channels
        c_hidden: Hidden dimension per head
        num_heads: Number of attention heads
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int = 32, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads

        self.msa_layer_norm = LayerNorm(c_m, affine=True)
        self.pair_layer_norm = LayerNorm(c_z, affine=True)

        # Linear layer to project pair representation to bias
        self.feat_2d_proj = nn.Linear(c_z, num_heads, bias=False)

        self.attention = MLXAttention(
            c_q=c_m,
            c_kv=c_m,
            c_hidden=c_hidden,
            num_heads=num_heads,
            c_out=c_m,
            gating=True
        )

    def __call__(
        self,
        msa_act: mx.array,
        pair_act: mx.array,
        msa_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass.

        Args:
            msa_act: MSA activations [N_seq, N_res, c_m]
            pair_act: Pair activations [N_res, N_res, c_z]
            msa_mask: MSA mask [N_seq, N_res]

        Returns:
            Updated MSA activations [N_seq, N_res, c_m]
        """
        N_seq = msa_act.shape[0]

        # Layer norms
        msa_act = self.msa_layer_norm(msa_act)
        pair_act = self.pair_layer_norm(pair_act)

        # Project pair representation to attention bias
        # [N_res, N_res, c_z] -> [N_res, N_res, num_heads]
        pair_bias = self.feat_2d_proj(pair_act)
        # Transpose to [num_heads, N_res, N_res]
        pair_bias = mx.transpose(pair_bias, (2, 0, 1))
        # Broadcast to batch: [N_seq, num_heads, N_res, N_res]
        pair_bias = mx.broadcast_to(
            mx.expand_dims(pair_bias, axis=0),
            (N_seq, self.num_heads, pair_act.shape[0], pair_act.shape[1])
        )

        # Create mask bias and add to pair_bias if provided
        if msa_mask is not None:
            mask_bias = (1e9 * (msa_mask - 1.0))
            # Expand to [N_seq, 1, 1, N_res] for broadcasting
            mask_bias = mx.expand_dims(mx.expand_dims(mask_bias, axis=1), axis=1)  # [N_seq, 1, 1, N_res]
            # Add mask to bias
            pair_bias = pair_bias + mask_bias

        # Apply attention
        output = self.attention(msa_act, msa_act, bias=pair_bias, mask=None)

        return output


class MSAColumnAttention(nn.Module):
    """
    MSA per-column attention.

    Jumper et al. (2021) Suppl. Alg. 8 "MSAColumnAttention"

    Args:
        c_m: MSA representation channels
        c_hidden: Hidden dimension per head
        num_heads: Number of attention heads
    """

    def __init__(self, c_m: int, c_hidden: int = 32, num_heads: int = 8):
        super().__init__()

        self.layer_norm = LayerNorm(c_m, affine=True)
        self.attention = MLXAttention(
            c_q=c_m,
            c_kv=c_m,
            c_hidden=c_hidden,
            num_heads=num_heads,
            c_out=c_m,
            gating=True
        )

    def __call__(self, msa_act: mx.array, msa_mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass.

        Args:
            msa_act: MSA activations [N_seq, N_res, c_m]
            msa_mask: MSA mask [N_seq, N_res]

        Returns:
            Updated MSA activations [N_seq, N_res, c_m]
        """
        # Swap to per-column: [N_res, N_seq, c_m]
        msa_act = mx.transpose(msa_act, (1, 0, 2))
        if msa_mask is not None:
            msa_mask = mx.transpose(msa_mask, (1, 0))

        # Layer norm
        msa_act = self.layer_norm(msa_act)

        # Create mask bias if provided
        if msa_mask is not None:
            mask_bias = (1e9 * (msa_mask - 1.0))
            # Expand to [N_res, 1, 1, N_seq] for broadcasting with attention
            mask_bias = mx.expand_dims(mx.expand_dims(mask_bias, axis=1), axis=1)  # [N_res, 1, 1, N_seq]
        else:
            mask_bias = None

        # Apply attention
        msa_act = self.attention(msa_act, msa_act, bias=mask_bias, mask=None)

        # Swap back to [N_seq, N_res, c_m]
        msa_act = mx.transpose(msa_act, (1, 0, 2))

        return msa_act


class OuterProductMean(nn.Module):
    """
    Outer product mean for updating pair representation from MSA.

    Jumper et al. (2021) Suppl. Alg. 10 "OuterProductMean"

    Args:
        c_m: MSA representation channels
        c_z: Pair representation channels
        c_hidden: Hidden channels for outer product
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int = 32):
        super().__init__()
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(c_m, affine=True)
        self.left_projection = nn.Linear(c_m, c_hidden, bias=False)
        self.right_projection = nn.Linear(c_m, c_hidden, bias=False)

        # Output projection from outer product
        self.output_projection = nn.Linear(c_hidden * c_hidden, c_z, bias=True)

    def __call__(self, msa_act: mx.array, msa_mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass.

        Args:
            msa_act: MSA activations [N_seq, N_res, c_m]
            msa_mask: MSA mask [N_seq, N_res]

        Returns:
            Pair representation update [N_res, N_res, c_z]
        """
        # Expand mask
        if msa_mask is not None:
            mask = mx.expand_dims(msa_mask, axis=-1)
        else:
            mask = 1.0

        # Layer norm
        act = self.layer_norm(msa_act)

        # Projections
        left_act = mask * self.left_projection(act)  # [N_seq, N_res, c_hidden]
        right_act = mask * self.right_projection(act)  # [N_seq, N_res, c_hidden]

        # Outer product mean: average over sequences
        # left_act: [N_seq, N_res_i, c_hidden]
        # right_act: [N_seq, N_res_j, c_hidden]
        # output: [N_res_i, N_res_j, c_hidden, c_hidden]

        # Use einsum for outer product: 'sic,sjd->sijcd'
        # Then flatten last two dims and mean over sequences
        outer = mx.einsum('sic,sjd->sijcd', left_act, right_act)

        # Mean over sequences
        outer = mx.mean(outer, axis=0)  # [N_res, N_res, c_hidden, c_hidden]

        # Flatten last two dimensions
        N_res = outer.shape[0]
        outer = mx.reshape(outer, (N_res, N_res, self.c_hidden * self.c_hidden))

        # Output projection
        output = self.output_projection(outer)

        return output


class EvoformerIteration(nn.Module):
    """
    Single iteration (block) of Evoformer stack.

    Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack" lines 2-10

    Args:
        c_m: MSA representation channels
        c_z: Pair representation channels
        c_hidden_msa_att: Hidden dimension for MSA attention
        c_hidden_opm: Hidden dimension for outer product mean
        c_hidden_tri_mul: Hidden dimension for triangle multiplication
        c_hidden_tri_att: Hidden dimension for triangle attention
        num_heads_msa: Number of heads for MSA attention
        num_heads_tri: Number of heads for triangle attention
        num_intermediate_factor: Factor for transition layer
    """

    def __init__(
        self,
        c_m: int = 256,
        c_z: int = 128,
        c_hidden_msa_att: int = 32,
        c_hidden_opm: int = 32,
        c_hidden_tri_mul: int = 128,
        c_hidden_tri_att: int = 32,
        num_heads_msa: int = 8,
        num_heads_tri: int = 4,
        num_intermediate_factor: int = 4
    ):
        super().__init__()

        # MSA processing
        self.msa_row_attention = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            num_heads=num_heads_msa
        )
        self.msa_column_attention = MSAColumnAttention(
            c_m=c_m,
            c_hidden=c_hidden_msa_att,
            num_heads=num_heads_msa
        )
        self.msa_transition = Transition(c_m, num_intermediate_factor)

        # Pair processing
        self.outer_product_mean = OuterProductMean(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_opm
        )

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_z=c_z,
            c_hidden=c_hidden_tri_mul,
            equation='ikc,jkc->ijc'
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_z=c_z,
            c_hidden=c_hidden_tri_mul,
            equation='kjc,kic->ijc'
        )

        self.triangle_attention_starting = TriangleAttention(
            c_z=c_z,
            c_hidden=c_hidden_tri_att,
            num_heads=num_heads_tri,
            orientation='per_row'
        )
        self.triangle_attention_ending = TriangleAttention(
            c_z=c_z,
            c_hidden=c_hidden_tri_att,
            num_heads=num_heads_tri,
            orientation='per_column'
        )

        self.pair_transition = Transition(c_z, num_intermediate_factor)

    def __call__(
        self,
        msa_act: mx.array,
        pair_act: mx.array,
        msa_mask: Optional[mx.array] = None,
        pair_mask: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass of single Evoformer iteration.

        Args:
            msa_act: MSA activations [N_seq, N_res, c_m]
            pair_act: Pair activations [N_res, N_res, c_z]
            msa_mask: MSA mask [N_seq, N_res]
            pair_mask: Pair mask [N_res, N_res]

        Returns:
            Updated (msa_act, pair_act)
        """
        # MSA processing
        msa_act = msa_act + self.msa_row_attention(msa_act, pair_act, msa_mask)
        msa_act = msa_act + self.msa_column_attention(msa_act, msa_mask)
        msa_act = msa_act + self.msa_transition(msa_act, mask=mx.expand_dims(msa_mask, axis=-1) if msa_mask is not None else None)

        # Outer product mean (updates pair representation from MSA)
        pair_act = pair_act + self.outer_product_mean(msa_act, msa_mask)

        # Pair processing
        pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act, pair_mask)
        pair_act = pair_act + self.triangle_multiplication_incoming(pair_act, pair_mask)
        pair_act = pair_act + self.triangle_attention_starting(pair_act, pair_mask)
        pair_act = pair_act + self.triangle_attention_ending(pair_act, pair_mask)
        pair_act = pair_act + self.pair_transition(pair_act, mask=mx.expand_dims(pair_mask, axis=-1) if pair_mask is not None else None)

        return msa_act, pair_act
