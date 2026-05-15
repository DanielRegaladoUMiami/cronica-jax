# ADR-001: Pivot from language modeling to data-to-text

**Date:** 2026-05-15
**Status:** Accepted

## Context

The original project plan was a decoder-only language model trained from scratch
on scraped Spanish-language football crónicas from underserved Latin American
second-division leagues (Primera Nacional, Liga de Expansión MX, etc.).

Three findings forced a rethink:

1. **`robots.txt` blocks commercial scraping.**
   Olé, Promiedos and Récord all disallow the relevant paths in `robots.txt`.
   Our scrapers correctly respect that and returned zero usable data from
   journalistic sources.

2. **Wikinoticias yielded a tiny corpus.**
   ~201 docs / ~93k BPE tokens. Diagnostic in `scripts/analyze_corpus.py`
   showed: for a 5M-param model, this corpus is **~1000× over-parameterized**
   (Chinchilla-optimal is ~5K params at this scale). The model would memorize,
   not generalize.

3. **The corpus content was wrong.**
   Top mentions were `Argentina, copa, mundial, selección` — i.e. World Cup /
   Copa América news, not 2nd-division narratives. The pitch ("crónicas de
   segundas divisiones LATAM") did not match the data.

## Decision

Pivot the project to **data-to-text**:

- Input: structured match statistics (teams, score, goals by minute,
  competition) from Kaggle's `davidcariboo/player-scores` (CC0).
- Target: synthetic Spanish crónicas generated via OpenAI `gpt-4o-mini` (or
  current mini-tier model) using **8 commentator-persona prompts**
  (rioplatense, mexicano, español, etc.) with public-facing descriptive labels
  (no real-name impersonation in the demo).
- Mini-LLM in pure JAX learns to map `<STATS>` → crónica.

This is a real task with a concrete product story (auto-generated match
reports), the data scale (~5k–50k pairs) is appropriate for a 5M-param model,
and the JAX-from-scratch technical core is unchanged.

## Consequences

**Kept (no changes):**
- `src/cronica/model.py` — pure JAX Transformer, RoPE/RMSNorm/SwiGLU.
- `src/cronica/tokenizer.py` — byte-level BPE.
- `src/cronica/sample.py` — sampling utilities.
- Infrastructure: GitHub repo, Apache 2.0, HF Space target.

**Modified:**
- Default model size: **25M → 5M params** (vocab=8k, d=256, 4 layers, 4 heads)
  to match the realistic corpus size.
- Training loop will add **loss masking** so cross-entropy only applies to
  the `<cronica>` block, not the `<STATS>` prompt.
- Tokenizer to re-train with special tokens
  `<stats>`, `</stats>`, `<cronica>`, `</cronica>`, plus style tokens.

**Added:**
- `scripts/load_kaggle.py` — download + filter rich matches.
- `scripts/build_prompts.py` — render structured `<STATS>` blocks.
- `scripts/synth_cronicas.py` — async OpenAI generation, 8 personas.

**Discarded:**
- `scripts/scrape_news.py` — kept in repo as a record but won't be re-run.
- `scripts/scrape_wikimedia.py` — archived; may revive as domain-adaptation
  pretrain in the future.

## Honest portfolio framing

The README will state the pivot openly. It demonstrates engineering judgment
(killing what's not working, picking a better problem with the same tools)
rather than hiding it.
