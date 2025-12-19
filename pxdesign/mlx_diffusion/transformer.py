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
MLX-accelerated DiffusionTransformer for hybrid PyTorch/MLX execution.

This module provides an MPS GPU-optimized implementation of the 16-block
DiffusionTransformer, which is the main computational bottleneck in diffusion sampling.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import numpy as np


class MLXLayerNorm(nn.Module):
    """MLX LayerNorm matching Protenix's implementation."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = mx.ones((normalized_shape,))
            self.bias = mx.zeros((normalized_shape,))

    def __call__(self, x: mx.array) -> mx.array:
        # Compute mean and variance over the last dimension
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        # Apply affine transformation if enabled
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm


class MLXLinear(nn.Module):
    """MLX Linear layer with bias option."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize weights
        scale = np.sqrt(1.0 / in_features)
        self.weight = mx.random.uniform(
            low=-scale, high=scale,
            shape=(out_features, in_features)
        )

        if bias:
            self.bias = mx.zeros((out_features,))

    def __call__(self, x: mx.array) -> mx.array:
        out = x @ self.weight.T
        if self.use_bias:
            out = out + self.bias
        return out


class MLXAttention(nn.Module):
    """
    Multi-head attention with pair bias for diffusion transformers.

    Implements the attention mechanism from Algorithm 24 in AlphaFold3.
    """

    def __init__(
        self,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        n_heads: int = 16,
    ):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.n_heads = n_heads
        self.c_hidden = c_a // n_heads

        assert c_a % n_heads == 0, f"c_a ({c_a}) must be divisible by n_heads ({n_heads})"

        # Q/K/V projections
        self.q_proj = MLXLinear(c_a, c_a, bias=False)
        self.k_proj = MLXLinear(c_a, c_a, bias=False)
        self.v_proj = MLXLinear(c_a, c_a, bias=False)

        # Output projection
        self.o_proj = MLXLinear(c_a, c_a, bias=True)

        # Gating
        self.gate_proj = MLXLinear(c_a, c_a, bias=True)

        # Pair bias projection
        self.bias_proj = MLXLinear(c_z, n_heads, bias=False)

        # LayerNorms
        self.ln_a = MLXLayerNorm(c_a)
        self.ln_z = MLXLayerNorm(c_z)

    def __call__(
        self,
        a: mx.array,  # [..., N_token, c_a]
        z: mx.array,  # [..., N_token, N_token, c_z]
    ) -> mx.array:
        """
        Args:
            a: Token representations [..., N_token, c_a]
            z: Pair representations [..., N_token, N_token, c_z]

        Returns:
            Updated token representations [..., N_token, c_a]
        """
        batch_shape = a.shape[:-2]
        N_token = a.shape[-2]

        # Normalize inputs
        a_norm = self.ln_a(a)

        # Project to Q, K, V
        q = self.q_proj(a_norm)  # [..., N_token, c_a]
        k = self.k_proj(a_norm)
        v = self.v_proj(a_norm)

        # Reshape for multi-head attention
        # [..., N_token, c_a] -> [..., N_token, n_heads, c_hidden] -> [..., n_heads, N_token, c_hidden]
        q = q.reshape(*batch_shape, N_token, self.n_heads, self.c_hidden)
        k = k.reshape(*batch_shape, N_token, self.n_heads, self.c_hidden)
        v = v.reshape(*batch_shape, N_token, self.n_heads, self.c_hidden)

        q = mx.transpose(q, axes=list(range(len(batch_shape))) + [len(batch_shape) + 1, len(batch_shape), len(batch_shape) + 2])
        k = mx.transpose(k, axes=list(range(len(batch_shape))) + [len(batch_shape) + 1, len(batch_shape), len(batch_shape) + 2])
        v = mx.transpose(v, axes=list(range(len(batch_shape))) + [len(batch_shape) + 1, len(batch_shape), len(batch_shape) + 2])

        # Compute attention bias from pair representation
        bias = self.bias_proj(self.ln_z(z))  # [..., N_token, N_token, n_heads]
        # [..., N_token, N_token, n_heads] -> [..., n_heads, N_token, N_token]
        bias = mx.transpose(bias, axes=list(range(len(batch_shape))) + [len(batch_shape) + 2, len(batch_shape), len(batch_shape) + 1])

        # Scaled dot-product attention
        # [..., n_heads, N_token, c_hidden] @ [..., n_heads, c_hidden, N_token] -> [..., n_heads, N_token, N_token]
        scale = 1.0 / mx.sqrt(mx.array(self.c_hidden, dtype=mx.float32))
        attn_scores = (q @ mx.transpose(k, axes=list(range(len(k.shape) - 2)) + [-1, -2])) * scale
        attn_scores = attn_scores + bias

        attn_weights = mx.softmax(attn_scores, axis=-1)

        # Apply attention to values
        # [..., n_heads, N_token, N_token] @ [..., n_heads, N_token, c_hidden] -> [..., n_heads, N_token, c_hidden]
        attn_out = attn_weights @ v

        # Reshape back
        # [..., n_heads, N_token, c_hidden] -> [..., N_token, n_heads, c_hidden] -> [..., N_token, c_a]
        attn_out = mx.transpose(attn_out, axes=list(range(len(batch_shape))) + [len(batch_shape) + 1, len(batch_shape), len(batch_shape) + 2])
        attn_out = attn_out.reshape(*batch_shape, N_token, self.c_a)

        # Gating
        gate = mx.sigmoid(self.gate_proj(a_norm))
        attn_out = gate * attn_out

        # Output projection
        out = self.o_proj(attn_out)

        return out


class MLXTransition(nn.Module):
    """
    Transition block with SwiGLU activation.

    Implements Algorithm 11 from AlphaFold3.
    """

    def __init__(self, c_in: int, n: int = 2):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.ln = MLXLayerNorm(c_in)
        self.linear_a = MLXLinear(c_in, n * c_in, bias=False)
        self.linear_b = MLXLinear(c_in, n * c_in, bias=False)
        self.linear_out = MLXLinear(n * c_in, c_in, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x_norm = self.ln(x)
        a = self.linear_a(x_norm)
        b = self.linear_b(x_norm)

        # SwiGLU: silu(a) * b
        out = nn.silu(a) * b
        out = self.linear_out(out)

        return out


class MLXAdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned on single representation."""

    def __init__(self, c_a: int = 768, c_s: int = 384):
        super().__init__()
        self.ln_a = MLXLayerNorm(c_a)
        self.ln_s = MLXLayerNorm(c_s)
        self.linear_scale = MLXLinear(c_s, c_a, bias=True)
        self.linear_shift = MLXLinear(c_s, c_a, bias=False)

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        a_norm = self.ln_a(a)
        s_norm = self.ln_s(s)

        scale = mx.sigmoid(self.linear_scale(s_norm))
        shift = self.linear_shift(s_norm)

        return scale * a_norm + shift


class MLXDiffusionTransformerBlock(nn.Module):
    """
    Single block of the diffusion transformer.

    Implements Algorithm 23 (one block) from AlphaFold3.
    """

    def __init__(
        self,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        n_heads: int = 16,
    ):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z

        # Adaptive LayerNorm for attention
        self.ada_ln = MLXAdaptiveLayerNorm(c_a, c_s)

        # Attention with pair bias
        self.attention = MLXAttention(c_a, c_s, c_z, n_heads)

        # Conditioned transition
        self.ada_ln_ff = MLXAdaptiveLayerNorm(c_a, c_s)
        self.transition = MLXTransition(c_a, n=2)

    def __call__(
        self,
        a: mx.array,  # [..., N_sample, N_token, c_a]
        s: mx.array,  # [..., N_sample, N_token, c_s]
        z: mx.array,  # [..., N_sample, N_token, N_token, c_z]
    ) -> mx.array:
        """
        Args:
            a: Token representations [..., N_sample, N_token, c_a]
            s: Conditioning single representations [..., N_sample, N_token, c_s]
            z: Pair representations [..., N_sample, N_token, N_token, c_z]

        Returns:
            Updated token representations [..., N_sample, N_token, c_a]
        """
        # Adaptive normalization + attention + residual
        a_norm = self.ada_ln(a, s)
        attn_out = self.attention(a_norm, z)
        a = a + attn_out

        # Adaptive normalization + feedforward + residual
        a_norm = self.ada_ln_ff(a, s)
        ff_out = self.transition(a_norm)
        a = a + ff_out

        return a


class MLXDiffusionTransformer(nn.Module):
    """
    Complete MLX-accelerated DiffusionTransformer (16 blocks).

    This is the main computational bottleneck in diffusion sampling.
    Implements Algorithm 23 from AlphaFold3.
    """

    def __init__(
        self,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 16,
        n_heads: int = 16,
    ):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.n_blocks = n_blocks
        self.n_heads = n_heads

        # Create transformer blocks
        self.blocks = [
            MLXDiffusionTransformerBlock(c_a, c_s, c_z, n_heads)
            for _ in range(n_blocks)
        ]

    def __call__(
        self,
        a: mx.array,  # [..., N_sample, N_token, c_a]
        s: mx.array,  # [..., N_sample, N_token, c_s]
        z: mx.array,  # [..., N_sample, N_token, N_token, c_z]
    ) -> mx.array:
        """
        Forward pass through all transformer blocks.

        Args:
            a: Token representations [..., N_sample, N_token, c_a]
            s: Conditioning single representations [..., N_sample, N_token, c_s]
            z: Pair representations [..., N_sample, N_token, N_token, c_z]

        Returns:
            Updated token representations [..., N_sample, N_token, c_a]
        """
        # Run through all blocks sequentially
        for i, block in enumerate(self.blocks):
            a = block(a, s, z)

        return a

    def load_from_pytorch(self, pytorch_model):
        """
        Load weights from a PyTorch DiffusionTransformer model.

        Args:
            pytorch_model: PyTorch DiffusionTransformer instance
        """
        import torch

        print(f"Loading {self.n_blocks} transformer blocks from PyTorch to MLX...")

        for i, (mlx_block, torch_block) in enumerate(zip(self.blocks, pytorch_model.blocks)):
            # Load attention weights
            mlx_block.attention.q_proj.weight = mx.array(
                torch_block.attention_pair_bias.attention.linear_q.weight.detach().cpu().numpy()
            )
            mlx_block.attention.k_proj.weight = mx.array(
                torch_block.attention_pair_bias.attention.linear_k.weight.detach().cpu().numpy()
            )
            mlx_block.attention.v_proj.weight = mx.array(
                torch_block.attention_pair_bias.attention.linear_v.weight.detach().cpu().numpy()
            )
            mlx_block.attention.o_proj.weight = mx.array(
                torch_block.attention_pair_bias.attention.linear_o.weight.detach().cpu().numpy()
            )
            mlx_block.attention.o_proj.bias = mx.array(
                torch_block.attention_pair_bias.attention.linear_o.bias.detach().cpu().numpy()
            )

            # Load gate projection
            mlx_block.attention.gate_proj.weight = mx.array(
                torch_block.attention_pair_bias.attention.linear_g.weight.detach().cpu().numpy()
            )
            mlx_block.attention.gate_proj.bias = mx.array(
                torch_block.attention_pair_bias.attention.linear_g.bias.detach().cpu().numpy()
            )

            # Load transition weights
            mlx_block.transition.linear_a.weight = mx.array(
                torch_block.conditioned_transition_block.transition.linear_no_bias_a.weight.detach().cpu().numpy()
            )
            mlx_block.transition.linear_b.weight = mx.array(
                torch_block.conditioned_transition_block.transition.linear_no_bias_b.weight.detach().cpu().numpy()
            )
            mlx_block.transition.linear_out.weight = mx.array(
                torch_block.conditioned_transition_block.transition.linear_no_bias.weight.detach().cpu().numpy()
            )

            if (i + 1) % 4 == 0:
                print(f"  Loaded blocks {i-2}-{i+1}/{self.n_blocks}")

        print(f"✓ Successfully loaded all {self.n_blocks} transformer blocks")
