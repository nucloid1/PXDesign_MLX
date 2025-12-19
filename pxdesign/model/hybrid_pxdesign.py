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
Hybrid ProtenixDesign with MLX-accelerated diffusion.

Provides automatic MLX acceleration for the diffusion transformer while
keeping the rest of the model in PyTorch for maximum compatibility.
"""

import logging

logger = logging.getLogger(__name__)


def enable_mlx_acceleration(model, use_mlx=None, verbose=True):
    """
    Enable MLX acceleration for the diffusion transformer in ProtenixDesign.

    This function modifies the model in-place by wrapping the diffusion_transformer
    component with a hybrid MLX/PyTorch version.

    Args:
        model: ProtenixDesign instance
        use_mlx: Force MLX (True), force PyTorch (False), or auto-detect (None)
        verbose: Print acceleration status

    Returns:
        The modified model (also modified in-place)
    """
    try:
        from pxdesign.mlx_diffusion import create_hybrid_diffusion_transformer

        # Check if model has diffusion_module
        if not hasattr(model, "diffusion_module"):
            if verbose:
                logger.info("Model has no diffusion_module, skipping MLX acceleration")
            return model

        # Check if diffusion_module has diffusion_transformer
        if not hasattr(model.diffusion_module, "diffusion_transformer"):
            if verbose:
                logger.info("DiffusionModule has no diffusion_transformer, skipping MLX acceleration")
            return model

        # Get the original PyTorch transformer
        pytorch_transformer = model.diffusion_module.diffusion_transformer

        # Create hybrid transformer
        hybrid_transformer = create_hybrid_diffusion_transformer(
            pytorch_transformer,
            use_mlx=use_mlx,
            verbose=verbose
        )

        # Replace the original transformer
        model.diffusion_module.diffusion_transformer = hybrid_transformer

        if verbose and hybrid_transformer.mlx_available:
            print("✓ MLX acceleration enabled for diffusion model")

        return model

    except ImportError as e:
        if verbose:
            logger.info(f"MLX acceleration not available: {e}")
        return model
    except Exception as e:
        if verbose:
            logger.warning(f"Failed to enable MLX acceleration: {e}")
        return model
