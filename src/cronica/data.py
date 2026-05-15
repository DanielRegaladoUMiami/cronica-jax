"""Data loader: HF dataset -> tokenized streaming batches for training."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def load_split(split: str = "train", repo: str = "DanielRegaladoCardoso/cronicas-futbol-latam"):
    """Load a parquet split from HF Hub."""
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    return ds


def load_local_split(path: Path):
    """Load a parquet file from disk (offline / Kaggle scenarios)."""
    from datasets import load_dataset

    return load_dataset("parquet", data_files=str(path), split="train")


def iter_token_batches(
    ds,
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    bos_id: int,
    eos_id: int,
    seed: int = 0,
) -> Iterator[np.ndarray]:
    """Pack tokenized documents into contiguous (B, T) int32 arrays.

    Strategy: concatenate `<bos> + tokens + <eos>` for each cronica, slide a
    window of length `seq_len + 1` (input + target).
    """
    rng = np.random.default_rng(seed)
    buffer: list[int] = []
    window = seq_len + 1
    target_buf = batch_size * window

    def docs():
        idxs = np.arange(len(ds))
        rng.shuffle(idxs)
        for i in idxs:
            yield ds[int(i)]["cronica"]

    for text in docs():
        ids = tokenizer.encode(text).ids if hasattr(tokenizer.encode(text), "ids") else tokenizer.encode(text)
        buffer.append(bos_id)
        buffer.extend(ids)
        buffer.append(eos_id)
        while len(buffer) >= target_buf:
            chunk = np.array(buffer[:target_buf], dtype=np.int32).reshape(batch_size, window)
            yield chunk
            buffer = buffer[target_buf:]
