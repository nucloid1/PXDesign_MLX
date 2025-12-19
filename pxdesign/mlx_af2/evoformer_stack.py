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
MLX implementation of complete Evoformer stack for AlphaFold2.

Based on Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack"
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from .evoformer import EvoformerIteration


class EvoformerStack(nn.Module):
    """
    Complete Evoformer stack with multiple iteration blocks.

    Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack"

    Args:
        num_blocks: Number of Evoformer iterations (typically 48)
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
        num_blocks: int = 48,
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
        self.num_blocks = num_blocks

        # Create list of Evoformer iterations
        self.blocks = [
            EvoformerIteration(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_tri_mul=c_hidden_tri_mul,
                c_hidden_tri_att=c_hidden_tri_att,
                num_heads_msa=num_heads_msa,
                num_heads_tri=num_heads_tri,
                num_intermediate_factor=num_intermediate_factor
            )
            for _ in range(num_blocks)
        ]

    def __call__(
        self,
        msa_act: mx.array,
        pair_act: mx.array,
        msa_mask: Optional[mx.array] = None,
        pair_mask: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass through complete Evoformer stack.

        Args:
            msa_act: MSA activations [N_seq, N_res, c_m]
            pair_act: Pair activations [N_res, N_res, c_z]
            msa_mask: MSA mask [N_seq, N_res]
            pair_mask: Pair mask [N_res, N_res]

        Returns:
            Updated (msa_act, pair_act)
        """
        # Apply each Evoformer block sequentially
        for i, block in enumerate(self.blocks):
            msa_act, pair_act = block(msa_act, pair_act, msa_mask, pair_mask)

            # Optional: print progress for debugging
            # if (i + 1) % 10 == 0:
            #     print(f"Completed {i+1}/{self.num_blocks} Evoformer blocks")

        return msa_act, pair_act


class RecyclingEmbedding(nn.Module):
    """
    Recycling embedding for iterative refinement.

    Args:
        c_m: MSA representation channels
        c_z: Pair representation channels
        min_bin: Minimum distance bin (Angstroms)
        max_bin: Maximum distance bin (Angstroms)
        num_bins: Number of distance bins
    """

    def __init__(
        self,
        c_m: int = 256,
        c_z: int = 128,
        min_bin: float = 3.25,
        max_bin: float = 20.75,
        num_bins: int = 15
    ):
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.num_bins = num_bins

        # Linear layers for recycling
        self.msa_recycle = nn.Linear(c_m, c_m, bias=True)
        self.pair_recycle = nn.Linear(c_z, c_z, bias=True)

        # Distance embedding for recycling
        self.dist_embedding = nn.Linear(num_bins, c_z, bias=False)

    def __call__(
        self,
        msa_act: mx.array,
        pair_act: mx.array,
        prev_positions: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        """
        Add recycling embeddings to MSA and pair representations.

        Args:
            msa_act: MSA activations [N_seq, N_res, c_m]
            pair_act: Pair activations [N_res, N_res, c_z]
            prev_positions: Previous iteration positions [N_res, 3] (optional)

        Returns:
            Updated (msa_act, pair_act)
        """
        # Add recycled MSA
        msa_act = msa_act + self.msa_recycle(msa_act)

        # Add recycled pair
        pair_act = pair_act + self.pair_recycle(pair_act)

        # Add distance-based recycling if positions are provided
        if prev_positions is not None:
            # Compute pairwise distances
            # prev_positions: [N_res, 3]
            diff = mx.expand_dims(prev_positions, axis=0) - mx.expand_dims(prev_positions, axis=1)
            # diff: [N_res, N_res, 3]
            distances = mx.sqrt(mx.sum(diff * diff, axis=-1))  # [N_res, N_res]

            # Create distance bins
            bin_edges = mx.linspace(self.min_bin, self.max_bin, self.num_bins + 1)
            # Digitize distances into bins
            dist_bins = mx.zeros((distances.shape[0], distances.shape[1], self.num_bins))
            for i in range(self.num_bins):
                lower = bin_edges[i]
                upper = bin_edges[i + 1] if i < self.num_bins - 1 else float('inf')
                mask = (distances >= lower) & (distances < upper)
                dist_bins[:, :, i] = mask.astype(mx.float32)

            # Embed distances and add to pair representation
            dist_embed = self.dist_embedding(dist_bins)
            pair_act = pair_act + dist_embed

        return msa_act, pair_act


class EvoformerWithRecycling(nn.Module):
    """
    Evoformer stack with recycling for iterative refinement.

    Args:
        num_recycles: Number of recycling iterations
        num_blocks: Number of Evoformer blocks per iteration
        ... (other args same as EvoformerStack)
    """

    def __init__(
        self,
        num_recycles: int = 3,
        num_blocks: int = 48,
        **kwargs
    ):
        super().__init__()
        self.num_recycles = num_recycles

        self.evoformer_stack = EvoformerStack(num_blocks=num_blocks, **kwargs)
        self.recycling_embedding = RecyclingEmbedding(
            c_m=kwargs.get('c_m', 256),
            c_z=kwargs.get('c_z', 128)
        )

    def __call__(
        self,
        msa_act: mx.array,
        pair_act: mx.array,
        msa_mask: Optional[mx.array] = None,
        pair_mask: Optional[mx.array] = None,
        prev_positions: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass with recycling.

        Args:
            msa_act: Initial MSA activations [N_seq, N_res, c_m]
            pair_act: Initial pair activations [N_res, N_res, c_z]
            msa_mask: MSA mask [N_seq, N_res]
            pair_mask: Pair mask [N_res, N_res]
            prev_positions: Previous positions for recycling [N_res, 3]

        Returns:
            Final (msa_act, pair_act)
        """
        for recycle_idx in range(self.num_recycles):
            # Add recycling embeddings
            if recycle_idx > 0:  # Skip recycling on first iteration
                msa_act, pair_act = self.recycling_embedding(
                    msa_act, pair_act, prev_positions
                )

            # Run Evoformer stack
            msa_act, pair_act = self.evoformer_stack(
                msa_act, pair_act, msa_mask, pair_mask
            )

        return msa_act, pair_act
