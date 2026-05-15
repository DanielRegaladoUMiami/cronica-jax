# ADR-004: KV cache for autoregressive inference

**Date:** 2026-05-15
**Status:** Implemented and tested

## Context

Initial inference path was the naive "recompute the full forward each step":

```python
for _ in range(max_new_tokens):
    logits = forward(params, tokens, cfg)   # O(T²) attention every step
    next_id = sample(logits[:, -1, :])
    tokens = concat([tokens, next_id])
```

On the HF Space CPU this gave ~450 ms/token (~68 s for 150 tokens). Each new
token recomputes attention over all prior tokens from scratch.

## Decision

Implement standard **KV cache** in pure JAX:

- `prefill(params, prompt_tokens, cfg) -> (logits, cache)` runs the full
  attention over the prompt once and returns per-layer K and V tensors.
- `decode_step(params, token, cfg, cache, pos) -> (logits, new_cache)` runs
  ONE attention step over the new token, appending its K and V to the
  cache. RoPE is applied at absolute position `pos`.

`sample.generate_cronica` now uses `prefill` + a `decode_step` loop instead
of repeated `forward` calls.

## Correctness

`tests/test_e2e.py::test_kv_cache_matches_full_forward` enforces token-by-
token equality: same prompt, same RNG seed → same generated tokens whether
we use the cached or uncached path. Any divergence fails the test loudly.

## Speedup

On the M4 CPU, measured 20-token decode after warmup:

| path     | wallclock | ms/token |
|----------|-----------|----------|
| uncached | ~10 s     | ~500 ms  |
| cached   | ~5 s      | ~250 ms  |

That's **~2× speedup** locally. On the Space CPU we expect similar or
slightly better. The remaining cost is JAX recompiling the JIT each step
because the cache grows by one along the time axis on every call — a
follow-up optimization would pre-allocate a fixed-length cache buffer with
`jax.lax.dynamic_update_slice` and `jit` the decode step, which would
unlock another 3–5× on top.

`tests/test_e2e.py::test_kv_cache_is_faster_than_uncached` enforces the
cached path stays at least 1.3× faster than the uncached path. If a future
change regresses the cache, the test fails.

## Files touched

- `src/cronica/model.py` — added `attention_step`, `block_step`,
  `prefill`, `decode_step`, `init_kv_cache`. Existing `forward` is
  unchanged and still used for training.
- `src/cronica/sample.py` — generation loop rewritten to use
  prefill + decode_step.
- `tests/test_e2e.py` — two new tests.

Forward / training paths are unchanged; training continues to use
`forward()` over the full batch with no cache.
