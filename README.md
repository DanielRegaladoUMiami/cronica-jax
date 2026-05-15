# cronica-jax

Mini-LLM (~25M params) written **from scratch in pure JAX** that generates Spanish-language post-match football reports for under-covered Latin American second-tier leagues (Ascenso MX, Primera Nacional Argentina, Primera B Colombia).

> **Status:** Work in progress — weekend build. README will be expanded with architecture diagrams, training curves, sample outputs, and a Hugging Face Space demo.

## Why this project

Top-flight football is over-covered. Second divisions, women's leagues, and youth tournaments routinely play full matches that never get a written report. This project explores whether a small, purpose-built language model can fill that gap — even if imperfectly.

Technical focus:
- **JAX from scratch**: Transformer, RoPE, RMSNorm, SwiGLU, training loop hand-written. Optax for AdamW.
- **TPU training**: `jax.pmap` data parallelism on Kaggle TPU v3-8.
- **End-to-end pipeline**: scrape → tokenize → train → sample → demo.

## Architecture (planned)

- Decoder-only Transformer, 6 layers, d_model=512, 8 heads, 1024 ctx, ~25M params.
- RoPE positional encoding, RMSNorm, SwiGLU MLP.
- BPE tokenizer (~16k vocab) trained on Spanish football corpus.

## Repo layout

```
src/cronica/    # model, tokenizer, data, train, sample
scripts/        # scrape and prepare data
notebooks/      # EDA
space/          # Gradio demo for HF Space
results/        # qualitative samples
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

[Daniel Regalado](https://github.com/DanielRegaladoUMiami) — MSBA, University of Miami.
