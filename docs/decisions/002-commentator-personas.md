# ADR-002: 8 commentator personas for synthetic data, descriptive labels in demo

**Date:** 2026-05-15
**Status:** Accepted

## Context

When generating synthetic crónicas with a large LLM (OpenAI `gpt-4o-mini`),
plain "write a Spanish football match report" prompts produce homogeneous,
neutral text. The training corpus loses the regional and stylistic variety
that real Spanish-language football commentary has.

## Decision

Use **8 distinct prompts**, each inspired by a real broadcaster's regional
voice and stylistic signature. In the **synthetic generation step** the
prompts reference the broadcasters by name (so the LLM picks up on style
cues). In the **public-facing demo** (HF Space) the same 8 styles are
exposed with **descriptive labels** — no real name surfaces to users.

| Internal persona (prompt only) | Public label                 | Region / Sabor                     |
|-------------------------------|-------------------------------|------------------------------------|
| Andrés Cantor                 | Rioplatense apasionado        | Argentina / ESPN, emocional, "GOOOL" |
| Mariano Closs                 | Rioplatense técnico           | Argentina / TyC, analítico         |
| Christian Martinoli           | Mexicano irreverente          | México / TV Azteca, sarcasmo       |
| Pablo Ramírez                 | Mexicano clásico              | México / Azteca, formal            |
| Fernando Palomo               | Centroamericano ESPN          | El Salvador / ESPN, pulido         |
| Manolo Lama                   | Español radiofónico           | España / COPE, tradicional         |
| Víctor Hugo Morales           | Rioplatense literario         | Uruguay, poético                   |
| Diego Latorre                 | Comentario técnico            | Argentina / ESPN, táctico          |

The JSON record for each synthetic crónica stores BOTH the internal persona
name (for reproducibility / debugging) and the public label. Only the public
label gets pushed to the dataset on Hugging Face Hub and to the Gradio demo
dropdown.

## Why this trade-off

- **Quality of training data:** using real-name references in the prompt lets
  the large LLM tap into its prior knowledge of these broadcasters' signature
  phrases, pacing, vocabulary. Output is more flavorful than "Spanish neutral
  with passionate tone."
- **Likeness risk:** publishing a demo where users select "Andrés Cantor"
  could be read as unauthorized impersonation. Descriptive labels in the
  public surface area sidestep that risk while keeping the stylistic
  diversity in the training signal.
- **Reproducibility:** keeping the internal mapping in the dataset metadata
  preserves the provenance for anyone who wants to verify or rebuild.

## Implementation

`scripts/synth_cronicas.py` defines `SYSTEM_PROMPTS` as a list of 8 dicts
with fields:

```python
{
  "persona_internal": "Andrés Cantor",     # only used to render the prompt
  "label_public":    "rioplatense_apasionado",
  "prompt":          "Eres Andrés Cantor, comentarista argentino..."
}
```

The output JSONL stores `label_public` and `style_token` (e.g.
`<style:rioplatense_apasionado>`) — that's the token the JAX model will
condition on during training. `persona_internal` is kept in a separate
internal CSV for reproducibility but is **not** pushed to the HF dataset.

## Distribution target

For 5k training pairs: ~625 per style, balanced. For 50k: same ratio
maintained via stratified sampling on the prompts step.
