# ADR-003: Future work — switch to encoder-decoder for better grounding

**Date:** 2026-05-15
**Status:** Proposed (held for v2)

## Context

The current cronica-jax model is **decoder-only** (GPT-style): a single
Transformer stack with causal self-attention, where the `<STATS>` block and
the target crónica live in the same sequence. The training is a standard
next-token language-modeling objective with loss masking over the prompt.

Held-out evaluation on 12 crónicas across 4 unseen matches shows a clear
two-part story:

- **Style fidelity is excellent** — 100% activation for the two distinctive
  styles (`rioplatense_apasionado`, `rioplatense_literario`), 50% for
  `comentario_tecnico`.
- **Grounding is poor** — 16.7% home-team recall, 25.0% away-team recall,
  0.0% literal-score recall, **0/39 scorer recall**.

The student model learned the *flavor* of each commentator persona but did
not learn to copy specific names, teams, or scores from the input stats.

## Why decoder-only struggles with grounding at this scale

To produce "Calafiori scored at 74'" in the crónica, the decoder must attend
backwards through the causal mask to find that token sequence in the
`<STATS>` portion. At a 5M-parameter scale with ~1.4M training tokens, the
model has not learned this copying pattern reliably. GPT-style models do
learn grounding eventually, but they typically need orders of magnitude
more data and parameters.

The original "Attention is All You Need" (Vaswani et al., 2017)
architecture solves exactly this problem with **cross-attention**: an
encoder produces contextualized embeddings of the source sequence (the
stats), and every decoder token attends explicitly to *all* encoder
positions. Copying a name from input to output becomes a one-step
attention operation rather than a deep backwards traversal through
self-attention.

## Decision (proposed for v2)

Reimplement cronica-jax as an **encoder-decoder Transformer** in the
spirit of the original Vaswani 2017 paper, while keeping the pure-JAX
hand-rolled philosophy.

Plan:

1. **Encoder**: 4-layer Transformer with bidirectional self-attention over
   the `<STATS>` block.
2. **Decoder**: 4-layer Transformer with
   (a) causal self-attention over the crónica being generated, and
   (b) **cross-attention** to the encoder output.
3. Reuse RoPE for self-attention, sinusoidal or learned for cross-attn.
4. Loss is computed only on decoder outputs; encoder gets gradient through
   the decoder's cross-attention.
5. Tokenizer and dataset are unchanged. We do not regenerate crónicas.

Estimated work: 3–4 hours of focused JAX coding plus a re-train.

## Why we are not doing this in v1

v1 deliberately uses decoder-only because:
1. It is the simpler implementation and shows the basic Transformer
   from-scratch effort that is the portfolio value.
2. It produces *something* end-to-end that we can publish today (dataset,
   model, Space, tests, metrics).
3. The honest comparison of v1 (decoder-only) vs v2 (encoder-decoder) is
   *itself* educational: it lets us empirically show why the original
   Vaswani architecture exists.

## Expected outcome of v2

Based on prior work in similar-size data-to-text models:

- Scorer recall should jump from ~0% to ~50–80%.
- Team-name recall should approach 90%+.
- Literal-score copying should become almost automatic.
- Style fidelity should be preserved (cross-attn does not erase style).

If v2 lands those numbers, the project becomes a **legitimate
data-to-text mini-LLM**, not just a from-scratch demonstrator.

## Other improvements stacked for v2 (less critical)

- **Larger model**: 15M–25M params (still small) — closes the
  param × data gap.
- **More training** (10k steps, ~4 epochs over the corpus).
- **Validation split + early stopping**: currently we have no held-in val,
  only held-out matches that were never tokenized.
