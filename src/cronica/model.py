"""Decoder-only Transformer in pure JAX.

No Flax, no Equinox — params are PyTree dicts of jnp.ndarray; the forward pass
is a plain functional `apply(params, tokens) -> logits`.

Architecture choices (Llama-flavor):
    - Pre-norm with RMSNorm (no LayerNorm)
    - Rotary position embeddings (RoPE) applied to Q and K
    - SwiGLU MLP
    - Tied input/output embeddings
    - Causal mask in attention

Sizes: configurable. Default ~25M params:
    vocab=16k, d_model=512, n_layers=6, n_heads=8, d_head=64, d_ff=1408, ctx=1024
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax


# ---------- config ----------


@dataclass(frozen=True)
class Config:
    vocab_size: int = 16000
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_head: int = 64       # d_model // n_heads
    d_ff: int = 1408       # ~= 8/3 * d_model rounded to multiple of 64
    max_seq_len: int = 1024
    rope_base: float = 10000.0

    @property
    def n_params_approx(self) -> int:
        c = self
        embed = c.vocab_size * c.d_model
        per_layer = (
            4 * c.d_model * c.d_model      # Wq, Wk, Wv, Wo (square approx)
            + 3 * c.d_model * c.d_ff       # gate, up, down in SwiGLU
            + 2 * c.d_model                # two RMSNorm weights
        )
        return embed + c.n_layers * per_layer + c.d_model  # +final RMSNorm


# ---------- param init ----------


def _trunc_normal(key, shape, std=0.02):
    return jax.random.truncated_normal(key, -2.0, 2.0, shape) * std


def init_params(cfg: Config, key) -> dict[str, Any]:
    keys = jax.random.split(key, 2 + cfg.n_layers * 7)
    ki = iter(keys)

    params: dict[str, Any] = {}
    params["tok_embed"] = _trunc_normal(next(ki), (cfg.vocab_size, cfg.d_model))
    params["final_norm"] = jnp.ones((cfg.d_model,))

    layers = []
    for _ in range(cfg.n_layers):
        layer = {
            "attn_norm": jnp.ones((cfg.d_model,)),
            "wq": _trunc_normal(next(ki), (cfg.d_model, cfg.n_heads * cfg.d_head)),
            "wk": _trunc_normal(next(ki), (cfg.d_model, cfg.n_heads * cfg.d_head)),
            "wv": _trunc_normal(next(ki), (cfg.d_model, cfg.n_heads * cfg.d_head)),
            "wo": _trunc_normal(next(ki), (cfg.n_heads * cfg.d_head, cfg.d_model)),
            "mlp_norm": jnp.ones((cfg.d_model,)),
            "w_gate": _trunc_normal(next(ki), (cfg.d_model, cfg.d_ff)),
            "w_up":   _trunc_normal(next(ki), (cfg.d_model, cfg.d_ff)),
            "w_down": _trunc_normal(next(ki), (cfg.d_ff, cfg.d_model)),
        }
        layers.append(layer)
    params["layers"] = layers
    return params


# ---------- core ops ----------


def rms_norm(x, weight, eps: float = 1e-5):
    var = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    inv = lax.rsqrt(var + eps).astype(x.dtype)
    return x * inv * weight


def _rope_freqs(d_head: int, seq_len: int, base: float, dtype=jnp.float32):
    """Precompute cos/sin frequencies for RoPE. Returns (cos, sin) of shape (T, d_head)."""
    half = d_head // 2
    freqs = 1.0 / (base ** (jnp.arange(0, half, dtype=dtype) / half))
    t = jnp.arange(seq_len, dtype=dtype)
    angles = jnp.einsum("i,j->ij", t, freqs)          # (T, half)
    cos = jnp.repeat(jnp.cos(angles), 2, axis=-1)     # (T, d_head)
    sin = jnp.repeat(jnp.sin(angles), 2, axis=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply RoPE to (B, H, T, d_head) tensor."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # interleave (-x2, x1)
    rotated = jnp.stack([-x2, x1], axis=-1).reshape(x.shape)
    cos = cos[None, None, :, :]   # broadcast over B, H
    sin = sin[None, None, :, :]
    return x * cos + rotated * sin


def swiglu(x_gate, x_up):
    return jax.nn.silu(x_gate) * x_up


def causal_attention(x, layer, cfg: Config, cos, sin):
    """Multi-head causal self-attention with RoPE."""
    B, T, D = x.shape
    H, dh = cfg.n_heads, cfg.d_head

    q = (x @ layer["wq"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)  # (B,H,T,dh)
    k = (x @ layer["wk"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    v = (x @ layer["wv"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    scale = 1.0 / jnp.sqrt(jnp.float32(dh))
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale     # (B,H,T,T)

    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    scores = jnp.where(mask[None, None, :, :], scores, -1e9)

    attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    out = jnp.einsum("bhts,bhsd->bhtd", attn, v)             # (B,H,T,dh)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, H * dh)    # (B,T,D)
    return out @ layer["wo"]


def block(x, layer, cfg: Config, cos, sin):
    h = rms_norm(x, layer["attn_norm"])
    x = x + causal_attention(h, layer, cfg, cos, sin)

    h = rms_norm(x, layer["mlp_norm"])
    gate = h @ layer["w_gate"]
    up = h @ layer["w_up"]
    ff = swiglu(gate, up) @ layer["w_down"]
    return x + ff


def forward(params, tokens, cfg: Config):
    """Forward pass: tokens (B, T) int32 -> logits (B, T, vocab_size).

    Tied embeddings: lm_head = tok_embed.T.
    """
    x = params["tok_embed"][tokens]  # (B, T, D)
    cos, sin = _rope_freqs(cfg.d_head, tokens.shape[1], cfg.rope_base, dtype=x.dtype)
    for layer in params["layers"]:
        x = block(x, layer, cfg, cos, sin)
    x = rms_norm(x, params["final_norm"])
    logits = x @ params["tok_embed"].T  # tied
    return logits


# ---------- utility ----------


def count_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(jnp.size(x) for x in leaves))
