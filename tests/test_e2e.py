"""End-to-end tests for cronica-jax. Hard asserts, no false positives.

Each test verifies one CONCRETE property and fails loudly otherwise.
Run with:  PYTHONPATH=src pytest -v tests/

The tests are intentionally tightly coupled to real artifacts (the tokenizer,
the checkpoint, the parquet dataset) — they are integration tests, not unit
tests. If an artifact is missing the test fails immediately.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

TOKENIZER_PATH = REPO_ROOT / "tokenizer.json"
CKPT_PATH = REPO_ROOT / "checkpoints/run01/ckpt_002000.pkl"
PAIRS_PATH = REPO_ROOT / "data/synthetic/pairs.jsonl"


# ============== DATA ==============


def test_pairs_file_exists_and_nonempty():
    assert PAIRS_PATH.exists(), f"FAIL: {PAIRS_PATH} not found"
    size = PAIRS_PATH.stat().st_size
    assert size > 1_000_000, f"FAIL: pairs.jsonl is too small ({size} bytes)"


def test_pairs_count_at_least_4000():
    with PAIRS_PATH.open() as f:
        n = sum(1 for _ in f)
    assert n >= 4000, f"FAIL: expected >=4000 pairs, got {n}"


def test_pairs_have_required_fields():
    required = {"match_id", "stats_block", "cronica", "style_label",
                "style_token", "model"}
    with PAIRS_PATH.open() as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            rec = json.loads(line)
            missing = required - set(rec)
            assert not missing, f"FAIL: line {i} missing fields {missing}"


def test_all_8_styles_present_and_balanced():
    EXPECTED = {
        "rioplatense_apasionado", "rioplatense_tecnico", "rioplatense_literario",
        "mexicano_irreverente", "mexicano_clasico",
        "centroamericano_espn", "espanol_radiofonico", "comentario_tecnico",
    }
    from collections import Counter
    counts = Counter()
    with PAIRS_PATH.open() as f:
        for line in f:
            counts[json.loads(line)["style_label"]] += 1
    assert set(counts) == EXPECTED, \
        f"FAIL: styles mismatch. Got {set(counts)} expected {EXPECTED}"
    mn, mx = min(counts.values()), max(counts.values())
    ratio = mx / mn
    assert ratio < 3.0, f"FAIL: style imbalance, ratio={ratio:.2f}"


def test_cronicas_are_in_spanish():
    """Heuristic: at least 80% of crónicas contain Spanish diacritics."""
    n_total = n_es = 0
    with PAIRS_PATH.open() as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            rec = json.loads(line)
            n_total += 1
            if any(c in rec["cronica"] for c in "áéíóúñ"):
                n_es += 1
    frac = n_es / n_total
    assert frac > 0.80, f"FAIL: only {frac:.1%} of crónicas contain Spanish diacritics"


def test_cronicas_length_reasonable():
    """No crónica should be shorter than 50 words (junk) or absurdly long."""
    short = long = 0
    n_total = 0
    with PAIRS_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            n = len(rec["cronica"].split())
            if n < 50:
                short += 1
            if n > 600:
                long += 1
            n_total += 1
    assert short < 0.05 * n_total, f"FAIL: {short}/{n_total} crónicas <50 words"
    assert long < 0.05 * n_total, f"FAIL: {long}/{n_total} crónicas >600 words"


# ============== TOKENIZER ==============


def _load_tokenizer():
    from tokenizers import Tokenizer
    assert TOKENIZER_PATH.exists(), f"FAIL: {TOKENIZER_PATH} missing"
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def test_tokenizer_vocab_size():
    tok = _load_tokenizer()
    n = len(tok.get_vocab())
    assert n >= 7000 and n <= 10000, f"FAIL: vocab size {n} outside [7000, 10000]"


def test_all_special_tokens_single_id():
    tok = _load_tokenizer()
    REQUIRED = [
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
    for t in REQUIRED:
        ids = tok.encode(t).ids
        assert len(ids) == 1, f"FAIL: token {t!r} -> {len(ids)} ids (expected 1)"


def test_tokenizer_zero_unk_on_corpus():
    tok = _load_tokenizer()
    unk_id = tok.get_vocab()["<unk>"]
    n_unk = 0
    n_tok = 0
    with PAIRS_PATH.open() as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            rec = json.loads(line)
            ids = tok.encode(rec["cronica"]).ids
            n_tok += len(ids)
            n_unk += sum(1 for x in ids if x == unk_id)
    assert n_unk == 0, f"FAIL: {n_unk} <unk> tokens in {n_tok} crónica tokens"


def test_tokenizer_roundtrip():
    tok = _load_tokenizer()
    sample = "Boca derrotó a River 2-1 con goles de Cavani y Benedetto."
    enc = tok.encode(sample)
    dec = tok.decode(enc.ids)
    # word count must roundtrip
    assert len(re.findall(r"\w+", sample)) == len(re.findall(r"\w+", dec)), \
        f"FAIL: roundtrip word count differs.\n  orig: {sample!r}\n  dec : {dec!r}"


# ============== MODEL ==============


@pytest.fixture(scope="module")
def model():
    """Load checkpoint once for all model tests."""
    from cronica.train import load_ckpt
    assert CKPT_PATH.exists(), f"FAIL: {CKPT_PATH} missing"
    params, cfg, step = load_ckpt(CKPT_PATH)
    return params, cfg, step


def test_checkpoint_loadable(model):
    params, cfg, step = model
    assert step == 2000, f"FAIL: expected step 2000, got {step}"


def test_param_count_close_to_5M(model):
    from cronica.model import count_params
    params, cfg, step = model
    n = count_params(params)
    assert 4_000_000 <= n <= 6_500_000, \
        f"FAIL: param count {n} outside [4M, 6.5M]"


def test_model_config_consistency(model):
    params, cfg, step = model
    assert cfg.vocab_size == 8000
    assert cfg.d_model == 256
    assert cfg.n_layers == 4
    assert cfg.n_heads == 4
    assert cfg.d_model % cfg.n_heads == 0


def test_forward_pass_finite(model):
    import jax.numpy as jnp
    from cronica.model import forward
    params, cfg, step = model
    tokens = jnp.zeros((1, 32), dtype=jnp.int32)
    logits = forward(params, tokens, cfg)
    assert logits.shape == (1, 32, cfg.vocab_size), \
        f"FAIL: logits shape {logits.shape}"
    assert bool(jnp.all(jnp.isfinite(logits))), \
        "FAIL: forward produced non-finite logits"


def test_forward_pass_no_nans_on_random_input(model):
    import jax
    import jax.numpy as jnp
    from cronica.model import forward
    params, cfg, step = model
    key = jax.random.PRNGKey(0)
    tokens = jax.random.randint(key, (2, 64), 0, cfg.vocab_size, dtype=jnp.int32)
    logits = forward(params, tokens, cfg)
    assert bool(jnp.all(jnp.isfinite(logits)))
    assert logits.shape == (2, 64, cfg.vocab_size)


# ============== SAMPLING ==============


def test_sampling_produces_text(model):
    from tokenizers import Tokenizer
    from cronica.sample import generate_cronica
    params, cfg, step = model
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    stats = (
        "<STATS>\nliga: La Liga\nlocal: Real Madrid\n"
        "visitante: Barcelona\nresultado: 2-1\n</STATS>"
    )
    text = generate_cronica(
        params, cfg, tok, stats, "rioplatense_apasionado",
        max_new_tokens=40, temperature=0.85, top_k=50, top_p=0.9, seed=0,
    )
    assert isinstance(text, str)
    assert len(text) > 30, f"FAIL: too few characters generated ({len(text)})"
    # Ensure no <special_token> leaked into the output
    for tok_str in ("<bos>", "<eos>", "<stats>", "</stats>", "<cronica>", "</cronica>"):
        assert tok_str not in text, f"FAIL: leaked special token {tok_str!r} in output"


def test_sampling_rejects_unknown_style(model):
    from tokenizers import Tokenizer
    from cronica.sample import generate_cronica
    params, cfg, step = model
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    stats = "<STATS>\nliga: X\nlocal: A\nvisitante: B\nresultado: 1-0\n</STATS>"
    with pytest.raises(ValueError):
        generate_cronica(
            params, cfg, tok, stats, "this-style-does-not-exist",
            max_new_tokens=10, seed=0,
        )


# ============== PIPELINE WIRING ==============


def test_data_encode_example_mask_only_on_target():
    """The target mask must be 1 only on cronica span, 0 on prompt + pad."""
    import numpy as np
    from tokenizers import Tokenizer
    from cronica.data import encode_example
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab = tok.get_vocab()
    rec = {
        "stats_block": "liga: La Liga\nresultado: 2-1",
        "style_token": "<style:rioplatense_apasionado>",
        "cronica": "¡Qué partido! Real Madrid se llevó la victoria.",
    }
    style_vocab = {k: v for k, v in vocab.items() if k.startswith("<style:")}
    tokens, mask = encode_example(
        rec, tok, seq_len=128,
        bos_id=vocab["<bos>"], eos_id=vocab["<eos>"], pad_id=vocab["<pad>"],
        stats_open=vocab["<stats>"], stats_close=vocab["</stats>"],
        cron_open=vocab["<cronica>"], cron_close=vocab["</cronica>"],
        style_vocab=style_vocab,
    )
    assert tokens.shape == (128,)
    assert mask.shape == (128,)
    # First token is bos with mask 0 (prompt)
    assert mask[0] == 0
    # First mask=1 must be after cron_open token
    cron_open = vocab["<cronica>"]
    if cron_open in tokens.tolist():
        idx = tokens.tolist().index(cron_open)
        assert mask[idx] == 0, "FAIL: <cronica> token itself should be masked OFF"
        assert mask[idx + 1] == 1, "FAIL: first cronica content token must be masked ON"
    # No mask=1 on pad positions
    pad_id = vocab["<pad>"]
    pad_positions = np.where(tokens == pad_id)[0]
    for p in pad_positions:
        assert mask[p] == 0, f"FAIL: pad token at pos {p} has mask=1"


def test_data_load_pairs():
    from cronica.data import load_pairs
    pairs = load_pairs(PAIRS_PATH)
    assert len(pairs) >= 4000
    p = pairs[0]
    assert "stats_block" in p and "cronica" in p and "style_token" in p
    assert p["style_token"].startswith("<style:")
