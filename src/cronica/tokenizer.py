"""BPE tokenizer trained on the cronicas corpus.

Wraps HuggingFace `tokenizers` with a thin API the rest of the package depends on.

CLI:
    python -m cronica.tokenizer train \
        --input data/clean/cronicas.jsonl \
        --out tokenizer.json \
        --vocab-size 16000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Sequence as NormSeq, Replace
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.processors import ByteLevel as ByteLevelProc
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    # Section delimiters for the data-to-text template
    "<stats>", "</stats>",
    "<cronica>", "</cronica>",
    # Style conditioning tokens (one per public commentator label)
    "<style:rioplatense_apasionado>",
    "<style:rioplatense_tecnico>",
    "<style:rioplatense_literario>",
    "<style:mexicano_irreverente>",
    "<style:mexicano_clasico>",
    "<style:centroamericano_espn>",
    "<style:espanol_radiofonico>",
    "<style:comentario_tecnico>",
]


def _iter_texts_from_jsonl(path: Path) -> Iterable[str]:
    """Yield text strings for tokenizer training.

    For the data-to-text pivot, we yield the FULL training example
    `<stats>STATS</stats> <style> <cronica>CRONICA</cronica>` so the BPE
    learns subwords that occur in both the prompt and the target.
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cronica = rec.get("cronica", "")
            stats = rec.get("stats_block", "")
            style = rec.get("style_token", "")
            if cronica and stats:
                # Render exactly as training format
                yield (
                    f"<stats>{stats}</stats>\n{style}\n<cronica>{cronica}</cronica>"
                )
            elif cronica:
                yield cronica


def train_tokenizer(
    input_path: Path,
    out_path: Path,
    vocab_size: int = 16000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the cronicas JSONL.

    Byte-level BPE handles Spanish diacritics and rare team/player names without
    UNK proliferation, and is the standard choice for GPT-style models.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NormSeq([NFC(), Replace(r"\s+", " ")])
    tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDec()
    tokenizer.post_processor = ByteLevelProc(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevelPre.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(_iter_texts_from_jsonl(input_path), trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def token_ids(tok_id_map: dict[str, int]) -> dict[str, int]:
    """Map of special token name -> id, for use in training/sampling code."""
    return {name: tok_id_map[name] for name in SPECIAL_TOKENS}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--input", type=Path, required=True)
    p_train.add_argument("--out", type=Path, default=Path("tokenizer.json"))
    p_train.add_argument("--vocab-size", type=int, default=16000)
    p_train.add_argument("--min-frequency", type=int, default=2)

    p_info = sub.add_parser("info")
    p_info.add_argument("--tokenizer", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        tok = train_tokenizer(args.input, args.out, args.vocab_size, args.min_frequency)
        vocab = tok.get_vocab()
        print(f"Trained tokenizer: vocab_size={len(vocab)} -> {args.out}")
        for s in SPECIAL_TOKENS:
            print(f"  {s}: {vocab[s]}")
    elif args.cmd == "info":
        tok = load_tokenizer(args.tokenizer)
        vocab = tok.get_vocab()
        print(f"vocab_size = {len(vocab)}")
        for s in SPECIAL_TOKENS:
            if s in vocab:
                print(f"  {s}: {vocab[s]}")
        sample = "Boca derrotó a River 2-1 en La Bombonera con goles de Cavani."
        enc = tok.encode(sample)
        print(f"\nSample: {sample!r}")
        print(f"IDs:    {enc.ids}")
        print(f"Tokens: {enc.tokens}")


if __name__ == "__main__":
    main()
