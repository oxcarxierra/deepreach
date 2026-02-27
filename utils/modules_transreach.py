from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.nn as nn


def _sdpa_double_backward_safe_context():
    """Force math SDPA backend to support higher-order autograd."""
    if not torch.cuda.is_available():
        return nullcontext()

    # PyTorch 2.0+ CUDA backend controls.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )

    # PyTorch API fallback.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(backends=[SDPBackend.MATH])
    except Exception:
        return nullcontext()


class WaveAct(nn.Module):
    """PINNsFormer-style wave activation: w1 * sin(x) + w2 * cos(x)."""

    def __init__(self):
        super().__init__()
        # Match original PINNsFormer implementation with global scalars.
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class PINNsformerFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Match original PINNsFormer FFN depth:
        # Linear -> WaveAct -> Linear -> WaveAct -> Linear
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TransReachEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = PINNsformerFFN(d_model=d_model, d_ff=d_ff)
        # Match original PINNsFormer pre-activation style around attn and FFN.
        self.act1 = WaveAct()
        self.act2 = WaveAct()
        self.dropout = nn.Dropout(dropout)
        # LayerNorm is kept optional for experimentation; default is disabled.
        self.norm1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.act1(x)
        with _sdpa_double_backward_safe_context():
            attn_out, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x2 = self.act2(x)
        ffn_out = self.ffn(x2)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransReachDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = PINNsformerFFN(d_model=d_model, d_ff=d_ff)
        self.act1 = WaveAct()
        self.act2 = WaveAct()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query2 = self.act1(query)
        with _sdpa_double_backward_safe_context():
            attn_out, _ = self.cross_attn(
                query2, memory, memory, need_weights=False
            )
        query = self.norm1(query + self.dropout(attn_out))
        query2 = self.act2(query)
        ffn_out = self.ffn(query2)
        query = self.norm2(query + self.dropout(ffn_out))
        return query


class SpatioTemporalMixer(nn.Module):
    """Linear projection stack that mixes [t, x] inputs into token embeddings."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        # Match original PINNsFormer linear embedding stage.
        self.linear_emb = nn.Linear(in_features, hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_emb(x)


class TransReachBVPNet(nn.Module):
    """Sequence-based DeepReach model inspired by PINNsFormer.

    Input:
        model_input['coords']: [..., D]
    Output:
        {
            'model_in': [..., D]              # anchor token inputs (j=0)
            'model_out': [..., 1]             # anchor token output
            'seq_model_in': [..., K, D]       # pseudo-sequence inputs
            'seq_model_out': [..., K, 1]      # sequence outputs
        }
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden_features: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        ffn_expansion: int = 2,
        pseudo_steps: int = 5,
        pseudo_dt: float = 1e-3,
        tmax: float = 1.0,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        **kwargs,
    ):
        super().__init__()
        if hidden_features % num_heads != 0:
            raise ValueError(
                f"hidden_features ({hidden_features}) must be divisible by num_heads ({num_heads})"
            )
        if pseudo_steps < 1:
            raise ValueError("pseudo_steps must be >= 1")

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.pseudo_steps = pseudo_steps
        self.pseudo_dt = pseudo_dt
        self.tmax = tmax

        self.mixer = SpatioTemporalMixer(
            in_features=in_features, hidden_features=hidden_features
        )
        # Keep explicit step embeddings for pseudo-time index ordering.
        self.step_embedding = nn.Parameter(
            torch.zeros(1, pseudo_steps, hidden_features)
        )
        nn.init.normal_(self.step_embedding, std=0.02)
        self.encoder_out_act = WaveAct()
        self.decoder_out_act = WaveAct()

        d_ff = hidden_features * ffn_expansion
        self.encoder_layers = nn.ModuleList(
            [
                TransReachEncoderLayer(
                    d_model=hidden_features,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransReachDecoderLayer(
                    d_model=hidden_features,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_features, out_features)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        print(self)

    def _generate_pseudo_sequence(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N, D], where coords[:, 0] is time
        seq = coords.unsqueeze(1).repeat(1, self.pseudo_steps, 1)
        time_offsets = (
            torch.arange(self.pseudo_steps, device=coords.device, dtype=coords.dtype)
            * self.pseudo_dt
        )
        seq[..., 0] = torch.clamp(seq[..., 0] + time_offsets, max=self.tmax)
        return seq

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        coords_org = model_input["coords"].clone().detach()
        original_shape = coords_org.shape[:-1]
        input_dim = coords_org.shape[-1]

        coords_flat = coords_org.reshape(-1, input_dim)
        seq_coords = self._generate_pseudo_sequence(coords_flat)
        # Use a flat leaf tensor as the autograd anchor to avoid view-related grad issues.
        seq_coords_flat = seq_coords.reshape(-1, input_dim).clone().detach().requires_grad_(True)
        seq_coords = seq_coords_flat.reshape(-1, self.pseudo_steps, input_dim)

        # [N, K, D] -> [N, K, H]
        mixed_tokens = self.mixer(seq_coords)
        mixed_tokens = mixed_tokens + self.step_embedding[:, : self.pseudo_steps]

        memory = mixed_tokens
        for layer in self.encoder_layers:
            memory = layer(memory)
        memory = self.encoder_out_act(memory)

        decoded = mixed_tokens
        for layer in self.decoder_layers:
            decoded = layer(decoded, memory)
        decoded = self.decoder_out_act(decoded)

        seq_outputs_flat = self.output_layer(decoded)  # [N, K, 1]

        anchor_inputs_flat = seq_coords[:, 0, :]
        anchor_outputs_flat = seq_outputs_flat[:, 0, :]

        seq_model_in = seq_coords.reshape(*original_shape, self.pseudo_steps, input_dim)
        seq_model_out = seq_outputs_flat.reshape(*original_shape, self.pseudo_steps, self.out_features)
        seq_model_out_flat = seq_outputs_flat.reshape(-1, self.out_features)
        model_in = anchor_inputs_flat.reshape(*original_shape, input_dim)
        model_out = anchor_outputs_flat.reshape(*original_shape, self.out_features)

        return {
            "model_in": model_in,
            "model_out": model_out,
            "seq_model_in": seq_model_in,
            "seq_model_out": seq_model_out,
            # Provide exact tensors used in graph for stable autograd jacobians.
            "seq_model_in_flat": seq_coords_flat,
            "seq_model_out_flat": seq_model_out_flat,
        }
