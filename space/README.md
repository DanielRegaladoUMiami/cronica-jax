---
title: cronica-jax
emoji: ⚽
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: 5M-param JAX mini-LLM. Stats → Spanish crónica.
---

# cronica-jax (HF Space)

Demo del mini-LLM de 5.26M parámetros escrito desde cero en JAX puro.
Inputs structured football match `<STATS>` block + commentator style →
produces a Spanish-language crónica.

The model and tokenizer are auto-downloaded from
[`DanielRegaladoCardoso/cronica-jax-5m`](https://huggingface.co/DanielRegaladoCardoso/cronica-jax-5m).

**Source**: https://github.com/DanielRegaladoUMiami/cronica-jax
