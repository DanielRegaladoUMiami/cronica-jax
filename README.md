# cronica-jax

> Mini-LLM (~5M params) written **from scratch in pure JAX** that learns
> **data-to-text**: take structured football match stats, produce a Spanish
> match report (crónica) in one of eight regional commentator styles.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## What this is

You give it structured stats:

```
<STATS>
liga: La Liga
fecha: 2023-04-15
local: Real Madrid
visitante: Barcelona
resultado: 2-1
goles:
  - 23' Vinícius Jr. (Real Madrid)
  - 45' Lewandowski (Barcelona)
  - 78' Bellingham (Real Madrid)
estadio: Santiago Bernabéu
</STATS>
```

It writes back a crónica in your chosen style (rioplatense apasionado,
mexicano irreverente, español radiofónico, etc.).

## Why

Top-flight football is over-covered. Mid- and lower-tier matches in dozens
of leagues play every week without a written report. This project explores
whether a small, purpose-built data-to-text model can fill that gap — while
also being a real piece of engineering: a Transformer hand-rolled in pure
JAX, eight regional commentator styles, training on Kaggle + Mac.

## Architecture

- **Decoder-only Transformer, ~5M params.** vocab=8k (byte-level BPE),
  d_model=256, 4 layers, 4 heads, max_seq_len=512.
- **Pure JAX.** `params` are PyTree dicts of `jnp.ndarray`. The forward pass
  is a plain `apply(params, tokens, cfg) -> logits`. No Flax, no Equinox.
- **Llama-flavor:** RoPE, RMSNorm pre-norm, SwiGLU MLP, tied embeddings.
- **Training:** Optax AdamW + cosine schedule + warmup + gradient clipping,
  with **loss masking** over the `<STATS>` prompt block.

## Data pipeline

```
Kaggle (davidcariboo/player-scores, CC0)
   └─ 82,073 matches with at least one goal
       └─ build_prompts.py renders <STATS> blocks in Spanish
           └─ synth_cronicas.py: OpenAI gpt-4o-mini, 8 commentator personas
               └─ pairs.jsonl  →  parquet  →  HF Hub
                   └─ JAX training (Mac M4 local + optional Kaggle TPU run)
                       └─ checkpoint → HF Hub model repo
                           └─ HF Space (Gradio demo)
```

See [`docs/decisions/`](docs/decisions/) for the full decision log
(pivot story, commentator personas trade-off).

## Eight commentator styles

| Public style label              | Region / flavor                         |
|---------------------------------|-----------------------------------------|
| `rioplatense_apasionado`        | Argentina, emocional, énfasis en goles  |
| `rioplatense_tecnico`           | Argentina, narrador analítico           |
| `mexicano_irreverente`          | México, sarcasmo, frases ingeniosas     |
| `mexicano_clasico`              | México, formal, neutro                  |
| `centroamericano_espn`          | El Salvador / ESPN Latam, pulido        |
| `espanol_radiofonico`           | España, cabina radial, tradicional      |
| `rioplatense_literario`         | Uruguay, prosa evocativa                |
| `comentario_tecnico`            | Análisis táctico                        |

(Synthetic training data was generated with the help of real-broadcaster
personas as prompts. Only descriptive labels are exposed in the demo. See
[`docs/decisions/002-commentator-personas.md`](docs/decisions/002-commentator-personas.md).)

## Repo layout

```
src/cronica/
  model.py       # Transformer in pure JAX
  tokenizer.py   # byte-level BPE
  train.py       # jit'd training loop, loss masking, Optax
  sample.py      # greedy / top-k / top-p generation
  data.py        # HF dataset loader + token batch iterator

scripts/
  load_kaggle.py     # download + filter rich matches
  build_prompts.py   # render <STATS> blocks in Spanish
  synth_cronicas.py  # async OpenAI generation, 8 personas
  clean.py           # quality + dedup filters
  build_dataset.py   # stratified parquet + HF push
  analyze_corpus.py  # diagnostic over any cleaned corpus

space/         # Gradio app for HF Space
notebooks/     # EDA, Kaggle training notebook
docs/decisions # architecture/decision records
results/       # sample outputs + training curves
```

## How to run end-to-end

```bash
# 1. Setup
pip install -e .

# 2. Download and filter Kaggle data (~3 min)
python -m scripts.load_kaggle --out data/kaggle

# 3. Build structured <STATS> prompts (~30 s for 50k matches)
python -m scripts.build_prompts \
  --in data/kaggle/matches_rich.parquet \
  --out data/prompts/prompts.jsonl --max 50000

# 4. Generate synthetic crónicas with OpenAI (cost ~$2 for 5k, ~$17 for 50k)
export OPENAI_API_KEY=...   # put it in ~/.zshrc, NOT here
python -m scripts.synth_cronicas \
  --in data/prompts/prompts.jsonl \
  --out data/synthetic/pairs.jsonl \
  --model gpt-4o-mini --max 5000 --concurrency 30

# 5. Train the byte-level BPE
python -m cronica.tokenizer train \
  --input data/synthetic/pairs.jsonl --out tokenizer.json --vocab-size 8000

# 6. Train the model (~10 min on Mac M4 for 5k pairs)
python -m cronica.train \
  --dataset-path data/clean \
  --tokenizer tokenizer.json \
  --out-dir checkpoints/run01

# 7. Sample
python -m cronica.sample \
  --ckpt checkpoints/run01/ckpt_005000.pkl \
  --tokenizer tokenizer.json \
  --prompt "$(cat examples/match_01.txt)"
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

Datasets:
- Source match data: `davidcariboo/player-scores` on Kaggle, CC0.
- Synthetic crónicas: generated by us with OpenAI, redistributed under
  Apache 2.0 with the synthetic dataset.

## Author

[Daniel Regalado](https://github.com/DanielRegaladoUMiami) — MSBA,
University of Miami. Built as a portfolio piece for ML/analytics roles.
