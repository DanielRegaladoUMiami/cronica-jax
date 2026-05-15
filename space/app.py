"""Gradio app for cronica-jax: stats -> Spanish crónica."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

# Make the `cronica` package importable on the Space.
SRC = Path(__file__).resolve().parent / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from tokenizers import Tokenizer  # noqa: E402

REPO = "DanielRegaladoCardoso/cronica-jax-5m"
CKPT_NAME = "ckpt_002000.pkl"

STYLES = [
    "rioplatense_apasionado",
    "rioplatense_tecnico",
    "rioplatense_literario",
    "mexicano_irreverente",
    "mexicano_clasico",
    "centroamericano_espn",
    "espanol_radiofonico",
    "comentario_tecnico",
]

EXAMPLE_STATS = """<STATS>
liga: La Liga
fecha: 2024-04-21
local: Real Madrid
visitante: Barcelona
resultado: 3-2
goles:
  - 18' Jude Bellingham (Real Madrid)
  - 30' Robert Lewandowski (Barcelona)
  - 56' Vinicius Junior (Real Madrid)
  - 71' Robert Lewandowski (Barcelona)
  - 89' Jude Bellingham (Real Madrid)
estadio: Santiago Bernabéu
asistencia: 78.412
árbitro: César Soto Grado
</STATS>"""


# Load model & tokenizer at startup (once).
print("Downloading checkpoint + tokenizer from HF Hub...")
tok_path = hf_hub_download(REPO, "tokenizer.json")
ckpt_path = hf_hub_download(REPO, CKPT_NAME)
print("Loading model...")

# Defer heavy imports until after we know files exist
from cronica.train import load_ckpt  # noqa: E402
from cronica.sample import generate_cronica  # noqa: E402

tok = Tokenizer.from_file(tok_path)
params, cfg, step = load_ckpt(Path(ckpt_path))
print(f"Loaded ckpt step={step}, vocab={cfg.vocab_size}, layers={cfg.n_layers}")


def generate(stats_block: str, style: str, temperature: float, top_p: float, max_tokens: int):
    if not stats_block.strip():
        return "Por favor pega un bloque <STATS>."
    if style not in STYLES:
        return f"Estilo desconocido: {style}"
    try:
        text = generate_cronica(
            params, cfg, tok,
            stats_block, style,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=50,
            top_p=float(top_p),
            seed=0,
        )
        return text or "(modelo no produjo texto)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


with gr.Blocks(title="cronica-jax — Stats → Crónica") as demo:
    gr.Markdown("""
# cronica-jax

**Mini-LLM (5.26M parámetros) escrito desde cero en JAX puro.**

Pegale los stats de un partido en el formato `<STATS>...</STATS>` y elegí
un estilo de comentarista. El modelo genera la crónica en español.

⚠️ Es un modelo deliberadamente pequeño (≈5M params) — la fluidez no compite
con un LLM grande. Su valor es la *craft*: implementación end-to-end del
Transformer desde primeros principios en JAX. Esperá texto coherente a
nivel de oración pero con hallucinations a nivel de párrafo.

[GitHub](https://github.com/DanielRegaladoUMiami/cronica-jax) ·
[Dataset](https://huggingface.co/datasets/DanielRegaladoCardoso/cronicas-d2t) ·
[Modelo](https://huggingface.co/DanielRegaladoCardoso/cronica-jax-5m)
""")

    with gr.Row():
        with gr.Column():
            stats = gr.Textbox(
                value=EXAMPLE_STATS, lines=15, max_lines=25,
                label="<STATS> block",
            )
            style = gr.Dropdown(STYLES, value=STYLES[0], label="Estilo de comentarista")
            with gr.Accordion("Parámetros de sampling", open=False):
                temperature = gr.Slider(0.3, 1.3, value=0.85, step=0.05, label="Temperature")
                top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p (nucleus)")
                max_tokens = gr.Slider(50, 300, value=150, step=10,
                                       label="Max new tokens (≥200 = lento en CPU)")
            btn = gr.Button("Generar crónica", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Crónica generada", lines=15, max_lines=25,
                             show_copy_button=True)

    btn.click(generate,
              inputs=[stats, style, temperature, top_p, max_tokens],
              outputs=out)

    gr.Markdown(f"Modelo cargado: `{REPO}` (step {step}, "
                f"{cfg.n_layers} layers, d_model={cfg.d_model})")

if __name__ == "__main__":
    demo.launch()
