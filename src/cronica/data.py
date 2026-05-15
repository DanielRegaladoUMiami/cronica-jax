"""Data loader for the data-to-text pivot.

Each training example is rendered as:

    <bos><stats>STATS_BLOCK</stats>\n<style:LABEL>\n<cronica>CRONICA_TEXT</cronica><eos>

During training, loss is only computed on the CRONICA portion (between
<cronica> and </cronica>, plus the closing </cronica> and <eos> tokens).
Tokens in the prompt (stats + style) are masked out so the model is only
penalized on what it must learn to produce.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np


def load_pairs(path: Path) -> list[dict]:
    """Load (stats_block, style_token, cronica) triples from JSONL."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "cronica" in rec and "stats_block" in rec:
                out.append(rec)
    return out


def encode_example(
    rec: dict, tokenizer, *,
    bos_id: int, eos_id: int,
    stats_open: int, stats_close: int,
    cron_open: int, cron_close: int,
    style_vocab: dict[str, int],
    seq_len: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (tokens, target_mask) of length seq_len.

    target_mask is 1 on positions where loss should be computed (the cronica
    body + closing tags) and 0 on prompt positions and padding.
    """
    stats_ids = tokenizer.encode(rec["stats_block"]).ids
    cron_ids = tokenizer.encode(rec["cronica"]).ids
    style_id = style_vocab.get(rec.get("style_token", ""), None)

    # Prompt (no loss): <bos><stats>STATS</stats><style>
    prompt = [bos_id, stats_open, *stats_ids, stats_close]
    if style_id is not None:
        prompt.append(style_id)
    prompt.append(cron_open)

    # Target (loss on): CRONICA</cronica><eos>
    target = [*cron_ids, cron_close, eos_id]

    seq = prompt + target
    mask = [0] * len(prompt) + [1] * len(target)

    # Truncate or pad to seq_len
    if len(seq) > seq_len:
        seq = seq[:seq_len]
        mask = mask[:seq_len]
    else:
        pad_n = seq_len - len(seq)
        seq = seq + [pad_id] * pad_n
        mask = mask + [0] * pad_n

    return np.asarray(seq, dtype=np.int32), np.asarray(mask, dtype=np.int32)


def iter_batches(
    pairs: list[dict],
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    bos_id: int, eos_id: int, pad_id: int,
    stats_open: int, stats_close: int,
    cron_open: int, cron_close: int,
    style_vocab: dict[str, int],
    seed: int = 0,
    shuffle: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (tokens, target_mask) batches of shape (B, T)."""
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(pairs))
    if shuffle:
        rng.shuffle(idxs)

    buf_tok: list[np.ndarray] = []
    buf_mask: list[np.ndarray] = []
    for i in idxs:
        tok, mask = encode_example(
            pairs[int(i)], tokenizer,
            bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            stats_open=stats_open, stats_close=stats_close,
            cron_open=cron_open, cron_close=cron_close,
            style_vocab=style_vocab,
            seq_len=seq_len,
        )
        buf_tok.append(tok)
        buf_mask.append(mask)
        if len(buf_tok) == batch_size:
            yield np.stack(buf_tok), np.stack(buf_mask)
            buf_tok = []
            buf_mask = []
    if buf_tok:
        yield np.stack(buf_tok), np.stack(buf_mask)
