"""Training loop for cronica-jax data-to-text.

Each example is `<bos><stats>...</stats><style:...><cronica>...</cronica><eos>`,
and we apply **loss masking** so cross-entropy is computed only on tokens
inside the <cronica>...</cronica> block (the targets the model must learn
to produce). Prompt tokens contribute zero loss.

Usage:
    python -m cronica.train \
        --pairs data/synthetic/pairs.jsonl \
        --tokenizer tokenizer.json \
        --out-dir checkpoints/run01 \
        --steps 2000 --batch-size 16 --seq-len 768
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from cronica.data import iter_batches, load_pairs
from cronica.model import Config, count_params, forward, init_params
from cronica.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


# ---------- masked loss ----------


def masked_ce_loss(params, batch_tokens, batch_mask, cfg: Config):
    """Cross-entropy with target masking.

    batch_tokens: (B, T+1) int32
    batch_mask:   (B, T+1) int32, 1 where loss applies
    """
    inputs = batch_tokens[:, :-1]
    targets = batch_tokens[:, 1:]
    mask = batch_mask[:, 1:].astype(jnp.float32)        # predict mask same as target

    logits = forward(params, inputs, cfg)               # (B, T, V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, cfg.vocab_size, dtype=log_probs.dtype)
    nll = -jnp.sum(one_hot * log_probs, axis=-1)        # (B, T)
    nll = nll * mask
    n_targets = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(nll) / n_targets


def make_lr_schedule(peak_lr: float, total_steps: int, warmup_steps: int):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=peak_lr * 0.1,
    )


def make_optimizer(peak_lr: float, total_steps: int, warmup_steps: int, weight_decay: float):
    schedule = make_lr_schedule(peak_lr, total_steps, warmup_steps)
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=weight_decay),
    )


def make_train_step(cfg: Config, optimizer):
    def step_fn(params, opt_state, tokens, mask):
        loss, grads = jax.value_and_grad(masked_ce_loss)(params, tokens, mask, cfg)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return jax.jit(step_fn)


# ---------- checkpoint IO ----------


def save_ckpt(out_dir: Path, step: int, params, cfg: Config) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ckpt_{step:06d}.pkl"
    arrays = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    with path.open("wb") as f:
        pickle.dump({"step": step, "params": arrays, "config": cfg.__dict__}, f)
    return path


def load_ckpt(path: Path):
    with path.open("rb") as f:
        blob = pickle.load(f)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), blob["params"])
    cfg = Config(**blob["config"])
    return params, cfg, blob["step"]


# ---------- driver ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path,
                        default=Path("data/synthetic/pairs.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/run01"))

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)

    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=704)

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trackio", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Devices: %s", jax.devices())

    cfg = Config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_model // args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
    )

    # Init model
    key = jax.random.PRNGKey(args.seed)
    init_key, _ = jax.random.split(key)
    params = init_params(cfg, init_key)
    logger.info("Params: %.2fM", count_params(params) / 1e6)

    # Optim
    optimizer = make_optimizer(args.peak_lr, args.steps, args.warmup_steps, args.weight_decay)
    opt_state = optimizer.init(params)
    step_fn = make_train_step(cfg, optimizer)

    # Tokenizer + data
    tok = load_tokenizer(args.tokenizer)
    vocab = tok.get_vocab()
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    pad_id = vocab["<pad>"]
    stats_open = vocab["<stats>"]
    stats_close = vocab["</stats>"]
    cron_open = vocab["<cronica>"]
    cron_close = vocab["</cronica>"]
    style_vocab = {k: v for k, v in vocab.items() if k.startswith("<style:")}

    pairs = load_pairs(args.pairs)
    logger.info("Loaded %d pairs", len(pairs))

    # Trackio (optional)
    tracker = None
    if args.trackio:
        try:
            import trackio
            tracker = trackio.init(project="cronica-jax", config=vars(args))
        except Exception as e:
            logger.warning("trackio init failed: %s", e)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps({
        **cfg.__dict__,
        "args": {k: str(v) for k, v in vars(args).items()},
    }, indent=2))

    # Training
    losses: list[float] = []
    t_start = time.time()
    step = 0
    while step < args.steps:
        batch_iter = iter_batches(
            pairs, tok,
            seq_len=args.seq_len + 1,  # +1 so masked_ce_loss has T inputs and T targets
            batch_size=args.batch_size,
            bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            stats_open=stats_open, stats_close=stats_close,
            cron_open=cron_open, cron_close=cron_close,
            style_vocab=style_vocab,
            seed=args.seed + step,
            shuffle=True,
        )
        for tokens, mask in batch_iter:
            step += 1
            tokens_j = jnp.asarray(tokens)
            mask_j = jnp.asarray(mask)
            params, opt_state, loss = step_fn(params, opt_state, tokens_j, mask_j)
            loss_val = float(loss)
            losses.append(loss_val)

            if step % args.log_every == 0:
                window = losses[-args.log_every:]
                avg = sum(window) / len(window)
                elapsed = time.time() - t_start
                toks_per_s = step * args.batch_size * args.seq_len / elapsed
                logger.info("step %5d | loss %.4f (avg %.4f) | %.0f tok/s",
                            step, loss_val, avg, toks_per_s)
                if tracker is not None:
                    tracker.log({"loss": loss_val, "loss_avg": avg,
                                 "toks_per_s": toks_per_s}, step=step)

            if step % args.ckpt_every == 0 or step >= args.steps:
                path = save_ckpt(args.out_dir, step, params, cfg)
                logger.info("Saved checkpoint: %s", path)

            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
