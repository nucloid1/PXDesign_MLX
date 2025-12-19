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
MLX-accelerated diffusion module for PXDesign.

This package provides Apple Silicon MPS GPU-optimized implementations of the
diffusion-based protein design model using MLX framework.

The hybrid approach accelerates the DiffusionTransformer (16 blocks) while
keeping other components in PyTorch for compatibility.
"""

from .hybrid_diffusion import HybridDiffusionTransformer, create_hybrid_diffusion_transformer
from .bridge import torch_to_mlx, mlx_to_torch, HybridExecutor

__all__ = [
    "HybridDiffusionTransformer",
    "create_hybrid_diffusion_transformer",
    "torch_to_mlx",
    "mlx_to_torch",
    "HybridExecutor",
]
