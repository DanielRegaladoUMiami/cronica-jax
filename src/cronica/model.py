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
    """Default targets ~5M params, matched to ~100k-token corpus.

    Use larger sizes (d_model=512, n_layers=6) only if scaling up the data.
    """

    vocab_size: int = 8000
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_head: int = 64       # d_model // n_heads
    d_ff: int = 704        # ~= 8/3 * d_model rounded to multiple of 64
    max_seq_len: int = 512
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
    """Apply RoPE to (B, H, T, d_head) tensor. cos/sin shape (T, d_head)."""
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
    """Multi-head causal self-attention with RoPE. No cache (training/prefill).

    cos/sin shape: (T, d_head).
    """
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


def attention_step(x, layer, cfg: Config, cos, sin, k_cache, v_cache):
    """Single-token attention using KV cache.

    x:        (B, 1, D)               — only the new token
    cos/sin:  (1, d_head)              — rotary at the new position only
    k_cache:  (B, H, prev_len, dh)
    v_cache:  (B, H, prev_len, dh)
    Returns:  (out (B,1,D), new_k (B,H,prev_len+1,dh), new_v same)
    """
    B = x.shape[0]
    H, dh = cfg.n_heads, cfg.d_head

    q = (x @ layer["wq"]).reshape(B, 1, H, dh).transpose(0, 2, 1, 3)  # (B,H,1,dh)
    k_new = (x @ layer["wk"]).reshape(B, 1, H, dh).transpose(0, 2, 1, 3)
    v_new = (x @ layer["wv"]).reshape(B, 1, H, dh).transpose(0, 2, 1, 3)

    q = apply_rope(q, cos, sin)
    k_new = apply_rope(k_new, cos, sin)

    k_all = jnp.concatenate([k_cache, k_new], axis=2)        # (B,H,L+1,dh)
    v_all = jnp.concatenate([v_cache, v_new], axis=2)

    scale = 1.0 / jnp.sqrt(jnp.float32(dh))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_all) * scale  # (B,H,1,L+1)
    # All keys are <= current position, so no extra mask needed.

    attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v_all)         # (B,H,1,dh)
    out = out.transpose(0, 2, 1, 3).reshape(B, 1, H * dh)
    return out @ layer["wo"], k_all, v_all


def block(x, layer, cfg: Config, cos, sin):
    h = rms_norm(x, layer["attn_norm"])
    x = x + causal_attention(h, layer, cfg, cos, sin)

    h = rms_norm(x, layer["mlp_norm"])
    gate = h @ layer["w_gate"]
    up = h @ layer["w_up"]
    ff = swiglu(gate, up) @ layer["w_down"]
    return x + ff


def block_step(x, layer, cfg: Config, cos, sin, k_cache, v_cache):
    """Single-token block step using cache."""
    h = rms_norm(x, layer["attn_norm"])
    attn_out, k_new, v_new = attention_step(h, layer, cfg, cos, sin, k_cache, v_cache)
    x = x + attn_out

    h = rms_norm(x, layer["mlp_norm"])
    gate = h @ layer["w_gate"]
    up = h @ layer["w_up"]
    ff = swiglu(gate, up) @ layer["w_down"]
    return x + ff, k_new, v_new


def forward(params, tokens, cfg: Config):
    """Forward pass over a full sequence (training / prefill).

    tokens (B, T) int32 -> logits (B, T, vocab_size).
    """
    x = params["tok_embed"][tokens]  # (B, T, D)
    cos, sin = _rope_freqs(cfg.d_head, tokens.shape[1], cfg.rope_base, dtype=x.dtype)
    for layer in params["layers"]:
        x = block(x, layer, cfg, cos, sin)
    x = rms_norm(x, params["final_norm"])
    logits = x @ params["tok_embed"].T  # tied
    return logits


def init_kv_cache(params, batch_size: int, cfg: Config, dtype=jnp.float32):
    """Empty per-layer KV cache (length 0)."""
    H, dh = cfg.n_heads, cfg.d_head
    return [
        {
            "k": jnp.zeros((batch_size, H, 0, dh), dtype=dtype),
            "v": jnp.zeros((batch_size, H, 0, dh), dtype=dtype),
        }
        for _ in range(cfg.n_layers)
    ]


def prefill(params, tokens, cfg: Config):
    """Run the model over the prompt and return (logits, populated cache).

    tokens: (B, T_prompt) int32
    Returns:
      logits  (B, T_prompt, V) — logits for every prompt position
      cache:  list of per-layer {"k": (B,H,T_prompt,dh), "v": same}
    """
    B, T = tokens.shape
    H, dh = cfg.n_heads, cfg.d_head

    x = params["tok_embed"][tokens]
    cos, sin = _rope_freqs(cfg.d_head, T, cfg.rope_base, dtype=x.dtype)

    cache = []
    for layer in params["layers"]:
        # Same as causal_attention, but also return K and V for cache.
        h = rms_norm(x, layer["attn_norm"])
        q = (h @ layer["wq"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = (h @ layer["wk"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = (h @ layer["wv"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        scale = 1.0 / jnp.sqrt(jnp.float32(dh))
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * dh) @ layer["wo"]
        x = x + out

        # MLP
        h = rms_norm(x, layer["mlp_norm"])
        gate = h @ layer["w_gate"]
        up = h @ layer["w_up"]
        ff = swiglu(gate, up) @ layer["w_down"]
        x = x + ff

        cache.append({"k": k, "v": v})

    x = rms_norm(x, params["final_norm"])
    logits = x @ params["tok_embed"].T
    return logits, cache


def decode_step(params, token, cfg: Config, cache, pos: int):
    """Decode one new token given the KV cache up to `pos` (exclusive).

    token: (B, 1) int32 — the newest token
    cache: list of per-layer {"k": (B,H,pos,dh), "v": same}
    pos:   integer, the absolute position of `token` in the sequence
    Returns:
      logits      (B, 1, V) — logits for the new token
      new_cache:  list of per-layer with k/v lengthened by 1
    """
    x = params["tok_embed"][token]  # (B, 1, D)
    # RoPE at the single absolute position `pos`.
    cos_full, sin_full = _rope_freqs(cfg.d_head, pos + 1, cfg.rope_base, dtype=x.dtype)
    cos = cos_full[pos:pos + 1]   # (1, d_head)
    sin = sin_full[pos:pos + 1]

    new_cache = []
    for layer, layer_cache in zip(params["layers"], cache):
        x, k_new, v_new = block_step(x, layer, cfg, cos, sin,
                                     layer_cache["k"], layer_cache["v"])
        new_cache.append({"k": k_new, "v": v_new})

    x = rms_norm(x, params["final_norm"])
    logits = x @ params["tok_embed"].T
    return logits, new_cache


# ---------- utility ----------


def count_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(jnp.size(x) for x in leaves))
