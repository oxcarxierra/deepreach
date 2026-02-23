"""
Transformer-SIREN Hybrid Architecture for DeepReach

This module implements a Transformer-based variant of the DeepReach value function
approximator, where the standard softmax attention is replaced with SIREN's
sinusoidal activation: sin(omega_0 * QK^T / sqrt(d_k)) * V.

This is a NEW, EXPERIMENTAL architecture. The original modules.py is NOT modified.

Key idea: Treat each coordinate dimension of the input as a "token", project it
into a high-dimensional embedding, and apply multi-head self-attention with
sinusoidal activations across coordinate dimensions. This allows the model to
learn inter-dimensional relationships through attention while preserving SIREN's
periodic inductive bias.

Reference:
- Sitzmann et al., "Implicit Neural Representations with Periodic Activation
  Functions", NeurIPS 2020 (SIREN)
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017 (Transformer)
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Initialization helpers (SIREN-compatible, independent from modules.py)
# ============================================================================


def siren_uniform_(tensor, fan_in, omega_0=30.0):
    """SIREN-compatible uniform initialization for hidden layers."""
    with torch.no_grad():
        bound = np.sqrt(6.0 / fan_in) / omega_0
        tensor.uniform_(-bound, bound)


def siren_first_layer_uniform_(tensor, fan_in):
    """SIREN first-layer initialization: uniform(-1/fan_in, 1/fan_in)."""
    with torch.no_grad():
        tensor.uniform_(-1.0 / fan_in, 1.0 / fan_in)


# ============================================================================
# Core components
# ============================================================================


class SineAttention(nn.Module):
    """Multi-head attention with sinusoidal activation instead of softmax.

    Instead of:  Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    We compute:  Attention(Q,K,V) = sin(omega_0 * QK^T / sqrt(d_k)) V

    This preserves SIREN's periodic inductive bias within the attention
    mechanism. Note: attention weights can be negative and do NOT sum to 1.

    Args:
        d_model: Dimension of the model (embedding size).
        num_heads: Number of attention heads.
        omega_0: Frequency for the sinusoidal activation (default: 30.0).
    """

    def __init__(self, d_model, num_heads, omega_0=30.0):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.omega_0 = omega_0
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        for linear in [self.W_q, self.W_k, self.W_v, self.W_o]:
            siren_uniform_(
                linear.weight, fan_in=linear.weight.size(-1), omega_0=self.omega_0
            )
            if linear.bias is not None:
                siren_uniform_(
                    linear.bias, fan_in=linear.weight.size(-1), omega_0=self.omega_0
                )

    def forward(self, x):
        """
        Args:
            x: (B, S, d_model) where S = number of tokens (coordinate dims).

        Returns:
            (B, S, d_model)
        """
        B, S, _ = x.shape

        # Project to Q, K, V and reshape for multi-head
        Q = (
            self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        )  # (B, H, S, d_k)
        K = (
            self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        )  # (B, H, S, d_k)
        V = (
            self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        )  # (B, H, S, d_k)

        # Sinusoidal attention: sin(omega_0 * QK^T / sqrt(d_k))
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, S, S)
        attn_weights = torch.sin(self.omega_0 * attn_logits)  # (B, H, S, S)

        # Apply attention weights to values
        out = torch.matmul(attn_weights, V)  # (B, H, S, d_k)

        # Concatenate heads and project
        out = (
            out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        )  # (B, S, d_model)
        out = self.W_o(out)

        return out


class SirenFFN(nn.Module):
    """Feed-forward network with SIREN sinusoidal activation.

    FFN(x) = W2 * sin(omega_0 * (W1 * x + b1)) + b2

    Args:
        d_model: Input/output dimension.
        d_ff: Hidden dimension (typically 4 * d_model, but we keep it = d_model for compactness).
        omega_0: Frequency for the sinusoidal activation.
    """

    def __init__(self, d_model, d_ff=None, omega_0=30.0):
        super().__init__()
        if d_ff is None:
            d_ff = d_model
        self.omega_0 = omega_0
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self._init_weights()

    def _init_weights(self):
        siren_uniform_(
            self.linear1.weight,
            fan_in=self.linear1.weight.size(-1),
            omega_0=self.omega_0,
        )
        if self.linear1.bias is not None:
            siren_uniform_(
                self.linear1.bias,
                fan_in=self.linear1.weight.size(-1),
                omega_0=self.omega_0,
            )
        siren_uniform_(
            self.linear2.weight,
            fan_in=self.linear2.weight.size(-1),
            omega_0=self.omega_0,
        )
        if self.linear2.bias is not None:
            siren_uniform_(
                self.linear2.bias,
                fan_in=self.linear2.weight.size(-1),
                omega_0=self.omega_0,
            )

    def forward(self, x):
        return self.linear2(torch.sin(self.omega_0 * self.linear1(x)))


class SirenTransformerBlock(nn.Module):
    """A single Transformer block with SineAttention and SirenFFN.

    Uses pre-norm (LayerNorm before attention/FFN) with residual connections.

    Args:
        d_model: Dimension of the model.
        num_heads: Number of attention heads.
        omega_0: SIREN frequency parameter.
    """

    def __init__(self, d_model, num_heads, omega_0=30.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SineAttention(d_model, num_heads, omega_0=omega_0)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SirenFFN(d_model, d_ff=d_model, omega_0=omega_0)

    def forward(self, x):
        """
        Args:
            x: (B, S, d_model)
        Returns:
            (B, S, d_model)
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.norm1(x))
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Top-level model (drop-in replacement for SingleBVPNet)
# ============================================================================


class SirenTransformerBVPNet(nn.Module):
    """Transformer-SIREN hybrid network for boundary value problems.

    Drop-in replacement for SingleBVPNet. Takes the same input format
    {'coords': (B, D)} and returns {'model_in': (B, D), 'model_out': (B, 1)}.

    Architecture:
        1. SIREN first layer: each coordinate dimension is independently
           projected to hidden_features, producing (B, D, hidden_features).
        2. L Transformer blocks with SineAttention over the D "tokens".
        3. Aggregation: mean-pool over tokens → (B, hidden_features).
        4. Linear output layer → (B, out_features).

    Args:
        in_features: Number of input coordinate dimensions (e.g., state_dim + 1).
        out_features: Number of output features (typically 1 for value function).
        hidden_features: Embedding dimension per token (d_model).
        num_heads: Number of attention heads.
        num_layers: Number of Transformer blocks.
        omega_0: SIREN frequency parameter (default: 30.0).
    """

    def __init__(
        self,
        in_features=2,
        out_features=1,
        hidden_features=256,
        num_heads=4,
        num_layers=3,
        omega_0=30.0,
        **kwargs,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.omega_0 = omega_0

        # Ensure d_model is divisible by num_heads
        assert hidden_features % num_heads == 0, (
            f"hidden_features ({hidden_features}) must be divisible by num_heads ({num_heads})"
        )

        # --- SIREN first layer: project each coordinate dim to hidden_features ---
        # Input (B, D) → reshape to (B, D, 1) → project to (B, D, hidden_features)
        self.coord_embed = nn.Linear(1, hidden_features)
        # First-layer SIREN init
        siren_first_layer_uniform_(self.coord_embed.weight, fan_in=1)
        if self.coord_embed.bias is not None:
            siren_first_layer_uniform_(self.coord_embed.bias, fan_in=1)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList(
            [
                SirenTransformerBlock(
                    d_model=hidden_features, num_heads=num_heads, omega_0=omega_0
                )
                for _ in range(num_layers)
            ]
        )

        # --- Output head ---
        self.output_linear = nn.Linear(hidden_features, out_features)
        # Xavier init for the final linear layer (no activation follows)
        nn.init.xavier_normal_(self.output_linear.weight)
        if self.output_linear.bias is not None:
            nn.init.zeros_(self.output_linear.bias)

        print(self)

    def forward(self, model_input, params=None):
        """
        Args:
            model_input: dict with key 'coords' of shape (B, D).

        Returns:
            dict with 'model_in' (B, D) and 'model_out' (B, out_features).
        """
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enable gradient computation w.r.t. coordinates (needed for PDE loss)
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)

        B, D = coords_org.shape

        # Project each coordinate dimension independently: (B, D) → (B, D, 1) → (B, D, H)
        x = coords_org.unsqueeze(-1)  # (B, D, 1)
        x = torch.sin(
            self.omega_0 * self.coord_embed(x)
        )  # (B, D, H) — SIREN first layer

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, D, H)

        # Aggregate over tokens (coordinate dimensions) via mean pooling
        x = x.mean(dim=1)  # (B, H)

        # Output projection
        output = self.output_linear(x)  # (B, out_features)

        return {"model_in": coords_org, "model_out": output}
