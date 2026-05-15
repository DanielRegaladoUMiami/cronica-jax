"""Sampling utilities: greedy, top-k, top-p (nucleus).

Usage:
    python -m cronica.sample \
        --ckpt checkpoints/run01/ckpt_005000.pkl \
        --tokenizer tokenizer.json \
        --prompt "En un partido cerrado, Quilmes recibió a Atlanta..."
        --max-new-tokens 200 --top-p 0.9 --temperature 0.8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cronica.model import forward
from cronica.tokenizer import load_tokenizer
from cronica.train import load_ckpt


def _filter_top_k(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    top_vals, _ = jax.lax.top_k(logits, k)
    threshold = top_vals[..., -1:, None]  # keep dims for broadcast
    threshold = top_vals[..., -1:]
    return jnp.where(logits < threshold, -jnp.inf, logits)


def _filter_top_p(logits: jnp.ndarray, p: float) -> jnp.ndarray:
    if p >= 1.0:
        return logits
    sorted_idx = jnp.argsort(logits, axis=-1)[..., ::-1]
    sorted_logits = jnp.take_along_axis(logits, sorted_idx, axis=-1)
    probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumprobs = jnp.cumsum(probs, axis=-1)
    cutoff = cumprobs > p
    # Keep at least 1 token: shift right by 1.
    cutoff = jnp.concatenate(
        [jnp.zeros_like(cutoff[..., :1]), cutoff[..., :-1]], axis=-1
    )
    sorted_logits = jnp.where(cutoff, -jnp.inf, sorted_logits)
    # Unsort
    inv_idx = jnp.argsort(sorted_idx, axis=-1)
    return jnp.take_along_axis(sorted_logits, inv_idx, axis=-1)


def sample_next(logits, key, temperature: float, top_k: int, top_p: float):
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        logits = _filter_top_k(logits, top_k)
    if top_p < 1.0:
        logits = _filter_top_p(logits, top_p)
    return jax.random.categorical(key, logits, axis=-1)


def generate(
    params, cfg, tokenizer, prompt: str,
    *, max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: int = 0,
) -> str:
    vocab = tokenizer.get_vocab()
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]

    prompt_ids = [bos_id] + tokenizer.encode(prompt).ids
    tokens = jnp.asarray(prompt_ids, dtype=jnp.int32)[None, :]  # (1, T)
    key = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        # Truncate to model context if needed.
        ctx = tokens[:, -cfg.max_seq_len:]
        logits = forward(params, ctx, cfg)
        next_logits = logits[:, -1, :]                              # (1, V)
        key, subkey = jax.random.split(key)
        next_id = sample_next(next_logits, subkey, temperature, top_k, top_p)
        tokens = jnp.concatenate([tokens, next_id[:, None]], axis=1)
        if int(next_id[0]) == eos_id:
            break

    full_ids = [int(x) for x in tokens[0].tolist()]
    return tokenizer.decode(full_ids[1:])  # drop initial <bos>


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="En un partido cerrado, ")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    params, cfg, step = load_ckpt(args.ckpt)
    tok = load_tokenizer(args.tokenizer)
    print(f"# checkpoint step={step}, params loaded")
    out = generate(
        params, cfg, tok, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        seed=args.seed,
    )
    print(out)


if __name__ == "__main__":
    main()
