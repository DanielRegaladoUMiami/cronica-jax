"""Sampling: feed a <stats> block + style token, get a crónica back."""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from cronica.model import decode_step, prefill
from cronica.tokenizer import load_tokenizer
from cronica.train import load_ckpt


def _filter_top_k(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    top_vals, _ = jax.lax.top_k(logits, k)
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
    cutoff = jnp.concatenate(
        [jnp.zeros_like(cutoff[..., :1]), cutoff[..., :-1]], axis=-1
    )
    sorted_logits = jnp.where(cutoff, -jnp.inf, sorted_logits)
    inv_idx = jnp.argsort(sorted_idx, axis=-1)
    return jnp.take_along_axis(sorted_logits, inv_idx, axis=-1)


def sample_next(logits, key, temperature: float, top_k: int, top_p: float):
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        logits = _filter_top_k(logits, top_k)
    if top_p < 1.0:
        logits = _filter_top_p(logits, top_p)
    return jax.random.categorical(key, logits, axis=-1)


def generate_cronica(
    params, cfg, tokenizer,
    stats_block: str, style_label: str,
    *, max_new_tokens: int = 400,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: int = 0,
) -> str:
    """Build prompt and sample until </cronica> or <eos>."""
    vocab = tokenizer.get_vocab()
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    stats_open = vocab["<stats>"]
    stats_close = vocab["</stats>"]
    cron_open = vocab["<cronica>"]
    cron_close = vocab["</cronica>"]
    style_tok = vocab.get(f"<style:{style_label}>")
    if style_tok is None:
        raise ValueError(
            f"Unknown style_label: {style_label}. Known styles: "
            f"{[k for k in vocab if k.startswith('<style:')]}"
        )

    stats_ids = tokenizer.encode(stats_block).ids
    prompt = [bos_id, stats_open, *stats_ids, stats_close, style_tok, cron_open]
    prompt_tokens = jnp.asarray(prompt, dtype=jnp.int32)[None, :]   # (1, T_prompt)
    key = jax.random.PRNGKey(seed)

    # ---- Prefill: process the whole prompt once, populate KV cache ----
    logits, cache = prefill(params, prompt_tokens, cfg)
    next_logits = logits[:, -1, :]
    key, subkey = jax.random.split(key)
    next_id = sample_next(next_logits, subkey, temperature, top_k, top_p)
    generated = [int(next_id[0])]
    pos = prompt_tokens.shape[1]   # absolute index of the next token to emit

    # ---- Decode loop: one token at a time, O(1) per step ----
    for _ in range(max_new_tokens - 1):
        nid_int = int(next_id[0])
        if nid_int == cron_close or nid_int == eos_id:
            break
        logits, cache = decode_step(
            params, next_id[:, None], cfg, cache, pos
        )
        next_logits = logits[:, -1, :]
        key, subkey = jax.random.split(key)
        next_id = sample_next(next_logits, subkey, temperature, top_k, top_p)
        generated.append(int(next_id[0]))
        pos += 1

    # Drop trailing </cronica> or <eos> if present
    if generated and generated[-1] in (cron_close, eos_id):
        generated = generated[:-1]
    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True,
                        help="Path to a text file containing the <STATS> block.")
    parser.add_argument("--style", default="rioplatense_apasionado",
                        help="One of the 8 style labels (without the <style:> prefix).")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stats_block = args.stats.read_text(encoding="utf-8").strip()
    params, cfg, step = load_ckpt(args.ckpt)
    tok = load_tokenizer(args.tokenizer)
    print(f"# checkpoint step={step}")
    print(f"# style={args.style}")
    print()
    print(generate_cronica(
        params, cfg, tok, stats_block, args.style,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
