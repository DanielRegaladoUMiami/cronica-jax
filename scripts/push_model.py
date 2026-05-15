"""Push the trained JAX checkpoint + tokenizer to Hugging Face Hub.

Validates BEFORE upload:
  - checkpoint loads cleanly
  - param count matches advertised size
  - forward pass produces non-NaN logits
  - generate produces at least 20 non-empty tokens for a smoke prompt
  - tokenizer.json has all expected special tokens as single ids

Only after every check passes does it upload to the model repo.

Usage:
    python -m scripts.push_model \
        --ckpt checkpoints/run01/ckpt_002000.pkl \
        --tokenizer tokenizer.json \
        --repo DanielRegaladoCardoso/cronica-jax-5m \
        --private
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


REQUIRED_SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<stats>", "</stats>", "<cronica>", "</cronica>",
    "<style:rioplatense_apasionado>",
    "<style:rioplatense_tecnico>",
    "<style:rioplatense_literario>",
    "<style:mexicano_irreverente>",
    "<style:mexicano_clasico>",
    "<style:centroamericano_espn>",
    "<style:espanol_radiofonico>",
    "<style:comentario_tecnico>",
]


def validate_tokenizer(path: Path) -> Tokenizer:
    assert path.exists(), f"FAIL: tokenizer not found at {path}"
    tok = Tokenizer.from_file(str(path))
    vocab = tok.get_vocab()
    for t in REQUIRED_SPECIAL_TOKENS:
        ids = tok.encode(t).ids
        assert len(ids) == 1, (
            f"FAIL: special token {t!r} encodes to {len(ids)} ids "
            f"(expected single id)"
        )
        assert t in vocab, f"FAIL: special token {t!r} not in vocab"
    logger.info("Tokenizer OK: vocab=%d, 16 special tokens all single-id",
                len(vocab))
    return tok


def validate_checkpoint(ckpt_path: Path, expected_params_min: int = 4_000_000):
    """Load checkpoint, verify param count, forward pass, generation."""
    # Import here to defer JAX import cost during arg parsing
    from cronica.train import load_ckpt
    from cronica.model import forward, count_params
    from cronica.sample import generate_cronica

    assert ckpt_path.exists(), f"FAIL: checkpoint not found at {ckpt_path}"
    params, cfg, step = load_ckpt(ckpt_path)
    n = count_params(params)
    assert n >= expected_params_min, (
        f"FAIL: param count {n} below expected min {expected_params_min}"
    )
    logger.info("Checkpoint OK: step=%d, params=%.2fM, cfg=%s",
                step, n / 1e6, cfg)

    # Forward smoke test
    dummy = jnp.zeros((1, 32), dtype=jnp.int32)
    logits = forward(params, dummy, cfg)
    assert logits.shape == (1, 32, cfg.vocab_size), \
        f"FAIL: logits shape {logits.shape}"
    finite = bool(jnp.all(jnp.isfinite(logits)))
    assert finite, "FAIL: forward produced non-finite logits"
    logger.info("Forward pass OK: logits %s, all finite", logits.shape)

    return params, cfg, step


def generation_smoke_test(params, cfg, tok: Tokenizer,
                          min_output_tokens: int = 20) -> str:
    from cronica.sample import generate_cronica

    sample_stats = (
        "<STATS>\n"
        "liga: La Liga\n"
        "fecha: 2024-03-15\n"
        "local: Real Madrid\n"
        "visitante: Atletico Madrid\n"
        "resultado: 2-1\n"
        "goles:\n"
        "  - 23' Vinicius Junior (Real Madrid)\n"
        "  - 67' Antoine Griezmann (Atletico Madrid)\n"
        "  - 88' Jude Bellingham (Real Madrid)\n"
        "</STATS>"
    )
    text = generate_cronica(
        params, cfg, tok,
        sample_stats, "rioplatense_apasionado",
        max_new_tokens=80, temperature=0.8, top_k=50, top_p=0.9, seed=0,
    )
    n_tok = len(tok.encode(text).ids)
    assert n_tok >= min_output_tokens, (
        f"FAIL: generation produced only {n_tok} tokens "
        f"(expected >= {min_output_tokens}). Output: {text!r}"
    )
    logger.info("Generation smoke test OK: produced %d tokens for sample prompt",
                n_tok)
    return text


MODEL_CARD = """---
license: apache-2.0
language:
  - es
tags:
  - jax
  - pure-jax
  - from-scratch
  - data-to-text
  - football
  - soccer
  - spanish
library_name: jax
pipeline_tag: text-generation
---

# cronica-jax-5m

A **5.26M parameter decoder-only Transformer**, **hand-written in pure JAX**
(no Flax, no Equinox), that learns the **data-to-text** task: take a
structured football match `<STATS>` block and produce a Spanish-language
crónica in one of 8 regional commentator styles.

This model is a **portfolio piece** that demonstrates the craft of implementing
a Transformer from first principles in JAX. It is intentionally small and is
not a state-of-the-art language model.

- **Code:** https://github.com/DanielRegaladoUMiami/cronica-jax
- **Training data:** [DanielRegaladoCardoso/cronicas-d2t](https://huggingface.co/datasets/DanielRegaladoCardoso/cronicas-d2t)

## Architecture

- Decoder-only Transformer, 5.26M params
- vocab=8000 (byte-level BPE), d_model=256, n_layers=4, n_heads=4,
  d_head=64, d_ff=704, max_seq_len=768
- RoPE positional encoding, RMSNorm pre-norm, SwiGLU MLP, tied embeddings
- Loss masking: cross-entropy applied **only** to tokens inside the
  `<cronica>...</cronica>` span; prompt tokens contribute zero loss.

All forward-pass primitives — attention, RoPE, RMSNorm, SwiGLU, the
training step — are hand-implemented in pure JAX, as a deliberate exercise
in understanding the math.

## Training

- 5,000 (stats, crónica) pairs from
  [cronicas-d2t](https://huggingface.co/datasets/DanielRegaladoCardoso/cronicas-d2t)
- Optax AdamW, peak_lr=3e-4, cosine schedule, warmup_steps=100, weight_decay=0.1
- Gradient clipping global_norm=1.0
- 2,000 steps, batch_size=8, seq_len=768
- Trained on Apple M4 CPU in ~28 minutes (no GPU/TPU)
- Loss: 8.74 (step 25) → 2.80 (step 2000), perplexity ≈ 16.4

## How to use

```python
import jax
from tokenizers import Tokenizer
from cronica.train import load_ckpt
from cronica.sample import generate_cronica
from huggingface_hub import hf_hub_download

tok_path = hf_hub_download("DanielRegaladoCardoso/cronica-jax-5m", "tokenizer.json")
ckpt   = hf_hub_download("DanielRegaladoCardoso/cronica-jax-5m", "ckpt_002000.pkl")

tok = Tokenizer.from_file(tok_path)
params, cfg, step = load_ckpt(ckpt)

stats = ("<STATS>\\n"
         "liga: La Liga\\n"
         "fecha: 2024-03-15\\n"
         "local: Real Madrid\\n"
         "visitante: Atletico Madrid\\n"
         "resultado: 2-1\\n"
         "goles:\\n"
         "  - 23' Vinicius Junior (Real Madrid)\\n"
         "  - 67' Antoine Griezmann (Atletico Madrid)\\n"
         "  - 88' Jude Bellingham (Real Madrid)\\n"
         "</STATS>")

text = generate_cronica(params, cfg, tok, stats,
                       style_label="rioplatense_apasionado",
                       temperature=0.85, top_k=50, top_p=0.9,
                       max_new_tokens=300)
print(text)
```

## Style labels

| label                       | region / flavor                          |
|----------------------------|------------------------------------------|
| `rioplatense_apasionado`   | Argentina, emotive, "gol" lengthened     |
| `rioplatense_tecnico`      | Argentina, analytical                    |
| `rioplatense_literario`    | Uruguay, evocative prose                 |
| `mexicano_irreverente`     | Mexico, sarcastic                        |
| `mexicano_clasico`         | Mexico, formal                           |
| `centroamericano_espn`     | El Salvador / ESPN Latam, polished       |
| `espanol_radiofonico`      | Spain, radio broadcast style             |
| `comentario_tecnico`       | tactical analysis                        |

## Limitations and honest caveats

- **Small model.** 5M params on 1.4M training tokens (~0.28 tokens/param,
  ~1000× over-parameterized vs. Chinchilla-optimal). Expect grammatical
  but unpolished output; do not expect the prose quality of a billion-
  parameter model.
- **Hallucinated context.** Training crónicas were generated by `gpt-4o-mini`,
  which sometimes added manager names, stadium nicknames, or derby
  references not present in the `<STATS>`. Our small model can reproduce
  this hallucination tendency. Use with care if grounding matters.
- **Coverage bias.** Training matches are weighted toward big European
  leagues + CONMEBOL competitions. Smaller leagues are under-represented.
- **Style overlap.** Three of the eight styles (`rioplatense_apasionado`,
  `rioplatense_literario`, `comentario_tecnico`) have strong distinctive
  vocab (`gooool`, `barrilete`, `transiciones`); the other five share
  more neutral journalistic vocabulary.

## License

Apache 2.0. See [LICENSE](https://github.com/DanielRegaladoUMiami/cronica-jax/blob/main/LICENSE).
"""


def push(ckpt: Path, tokenizer: Path, card_text: str, repo: str, private: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        whoami = api.whoami()
    except Exception as e:
        raise SystemExit(f"FAIL: HF auth not configured. Run `hf auth login`. {e}")
    logger.info("HF auth OK as %s", whoami.get("name"))

    api.create_repo(repo_id=repo, repo_type="model",
                    exist_ok=True, private=private)

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(card_text)
        card_path = f.name

    api.upload_file(path_or_fileobj=str(ckpt),
                    path_in_repo=ckpt.name,
                    repo_id=repo, repo_type="model")
    api.upload_file(path_or_fileobj=str(tokenizer),
                    path_in_repo="tokenizer.json",
                    repo_id=repo, repo_type="model")
    api.upload_file(path_or_fileobj=card_path,
                    path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    logger.info("Pushed to https://huggingface.co/%s", repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--repo", default="DanielRegaladoCardoso/cronica-jax-5m")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tok = validate_tokenizer(args.tokenizer)
    params, cfg, step = validate_checkpoint(args.ckpt)
    sample_text = generation_smoke_test(params, cfg, tok)
    logger.info("Sample crónica preview: %s",
                sample_text[:200].replace("\n", " "))

    if args.dry_run:
        logger.info("DRY RUN: all validations passed. Skipping HF upload.")
        return

    push(args.ckpt, args.tokenizer, MODEL_CARD, args.repo, args.private)


if __name__ == "__main__":
    main()
