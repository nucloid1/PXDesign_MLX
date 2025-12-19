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
MLX-based AlphaFold2 implementation for Apple Silicon.

This package provides a hybrid JAX-MLX implementation of AlphaFold2
optimized for MPS GPU acceleration on M3 Max processors.
"""

__version__ = "0.1.0"

from .jax_mlx_bridge import jax_to_mlx, mlx_to_jax

__all__ = ["jax_to_mlx", "mlx_to_jax"]
