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
Complete MLX AlphaFold2 model for evaluation.

This module combines the Evoformer and Structure Module into a complete
end-to-end AlphaFold2 model optimized for MPS GPU on Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional

from .evoformer_stack import EvoformerStack
from .structure_module import StructureModule, QuatAffine


class MLXAlphaFold2(nn.Module):
    """
    Complete MLX AlphaFold2 model for structure prediction.

    This is a simplified version focused on evaluation (not training).
    It takes preprocessed features and outputs 3D coordinates.

    Args:
        c_m: MSA representation channels (default: 256)
        c_z: Pair representation channels (default: 128)
        c_s: Single (node) representation channels (default: 384)
        num_evoformer_blocks: Number of Evoformer iterations (default: 48)
        num_structure_iterations: Number of structure module iterations (default: 8)
        num_recycles: Number of recycling iterations (default: 3)
    """

    def __init__(
        self,
        c_m: int = 256,
        c_z: int = 128,
        c_s: int = 384,
        num_evoformer_blocks: int = 48,
        num_structure_iterations: int = 8,
        num_recycles: int = 3
    ):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_s = c_s
        self.num_recycles = num_recycles

        # Evoformer stack
        self.evoformer = EvoformerStack(
            num_blocks=num_evoformer_blocks,
            c_m=c_m,
            c_z=c_z
        )

        # Single representation projection from MSA
        # Take first sequence from MSA as single representation
        self.msa_to_single = nn.Linear(c_m, c_s, bias=True)

        # Structure module
        self.structure_module = StructureModule(
            num_iterations=num_structure_iterations,
            c_s=c_s,
            c_z=c_z
        )

    def __call__(
        self,
        msa_act: mx.array,
        pair_act: mx.array,
        msa_mask: Optional[mx.array] = None,
        pair_mask: Optional[mx.array] = None,
        num_recycles: Optional[int] = None
    ) -> Dict[str, mx.array]:
        """
        Forward pass.

        Args:
            msa_act: MSA activations [N_seq, N_res, c_m]
            pair_act: Pair activations [N_res, N_res, c_z]
            msa_mask: MSA mask [N_seq, N_res] (optional)
            pair_mask: Pair mask [N_res, N_res] (optional)
            num_recycles: Number of recycling iterations (overrides default)

        Returns:
            Dictionary containing:
                - 'final_atom_positions': [N_res, 3] backbone CA positions
                - 'quaternions': [N_res, 4] backbone orientations
                - 'translations': [N_res, 3] backbone translations
                - 'single': [N_res, c_s] final single representation
                - 'pair': [N_res, N_res, c_z] final pair representation
        """
        N_res = pair_act.shape[0]

        if num_recycles is None:
            num_recycles = self.num_recycles

        if msa_mask is None:
            msa_mask = mx.ones((msa_act.shape[0], N_res))
        if pair_mask is None:
            pair_mask = mx.ones((N_res, N_res))

        # Recycling loop
        prev_s = None
        prev_affine = None

        for recycle_idx in range(num_recycles + 1):  # +1 because we do one final pass
            # Run Evoformer
            msa_out, pair_out = self.evoformer(msa_act, pair_act, msa_mask, pair_mask)

            # Extract single representation from MSA (first sequence)
            single = msa_out[0]  # [N_res, c_m]
            single = self.msa_to_single(single)  # [N_res, c_s]

            # Add recycling from previous iteration
            if prev_s is not None and recycle_idx > 0:
                single = single + prev_s

            # Create residue mask for structure module
            residue_mask = mx.ones((N_res, 1))

            # Run structure module
            s_out, affine_out = self.structure_module(
                single,
                pair_out,
                residue_mask,
                initial_affine=prev_affine
            )

            # Store for next recycle
            prev_s = s_out
            prev_affine = affine_out

            # Don't recycle on the last iteration
            if recycle_idx == num_recycles:
                break

        # Extract final coordinates (CA positions from translations)
        final_positions = affine_out.translation

        return {
            'final_atom_positions': final_positions,
            'quaternions': affine_out.quaternion,
            'translations': affine_out.translation,
            'rotations': affine_out.rotation,
            'single': s_out,
            'pair': pair_out,
            'msa': msa_out
        }


def create_mlx_alphafold2(
    model_name: str = "model_1",
    num_recycles: int = 3,
    **kwargs
) -> MLXAlphaFold2:
    """
    Create an MLX AlphaFold2 model with standard configurations.

    Args:
        model_name: Name of the model (e.g., "model_1", "model_2", etc.)
        num_recycles: Number of recycling iterations
        **kwargs: Additional arguments for MLXAlphaFold2

    Returns:
        Initialized MLXAlphaFold2 model
    """
    # Standard AF2 configuration
    config = {
        'c_m': 256,
        'c_z': 128,
        'c_s': 384,
        'num_evoformer_blocks': 48,
        'num_structure_iterations': 8,
        'num_recycles': num_recycles
    }

    # Override with any provided kwargs
    config.update(kwargs)

    model = MLXAlphaFold2(**config)

    return model
