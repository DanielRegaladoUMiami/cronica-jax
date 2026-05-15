"""Training loop for cronica-jax.

Pure JAX + Optax. Designed to run on:
  - single-host CPU (smoke test on Mac)
  - single-host TPU v3-8 via pmap (Kaggle)
  - single GPU (Colab fallback)

Usage:
    python -m cronica.train \
        --dataset-path data/clean \
        --tokenizer tokenizer.json \
        --out-dir checkpoints/run01 \
        --steps 5000 --batch-size 64 --seq-len 1024
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

from cronica.data import iter_token_batches, load_local_split
from cronica.model import Config, count_params, forward, init_params
from cronica.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


# ---------- loss and step ----------


def cross_entropy_loss(params, batch, cfg: Config):
    """batch: (B, T+1) int32. Inputs = batch[:, :-1], targets = batch[:, 1:]."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = forward(params, inputs, cfg)                  # (B, T, V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, cfg.vocab_size, dtype=log_probs.dtype)
    nll = -jnp.sum(one_hot * log_probs, axis=-1)           # (B, T)
    return jnp.mean(nll)


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
    def step_fn(params, opt_state, batch):
        loss, grads = jax.value_and_grad(cross_entropy_loss)(params, batch, cfg)
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
    parser.add_argument("--dataset-path", type=Path, default=Path("data/clean"))
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/run01"))

    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)

    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=704)

    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trackio", action="store_true",
                        help="Log metrics to trackio if installed.")
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

    key = jax.random.PRNGKey(args.seed)
    init_key, _ = jax.random.split(key)
    params = init_params(cfg, init_key)
    logger.info("Params: %.2fM", count_params(params) / 1e6)

    optimizer = make_optimizer(args.peak_lr, args.steps, args.warmup_steps, args.weight_decay)
    opt_state = optimizer.init(params)
    step_fn = make_train_step(cfg, optimizer)

    # Data
    tok = load_tokenizer(args.tokenizer)
    vocab = tok.get_vocab()
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]

    train_ds = load_local_split(args.dataset_path / "train.parquet")
    batch_iter = iter_token_batches(
        train_ds, tok, seq_len=args.seq_len, batch_size=args.batch_size,
        bos_id=bos_id, eos_id=eos_id, seed=args.seed,
    )

    # Optional trackio
    tracker = None
    if args.trackio:
        try:
            import trackio
            tracker = trackio.init(project="cronica-jax", config=vars(args))
        except Exception as e:
            logger.warning("trackio init failed: %s", e)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(
        json.dumps({**cfg.__dict__, "args": {k: str(v) for k, v in vars(args).items()}},
                   indent=2),
    )

    losses: list[float] = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(batch_iter)
        except StopIteration:
            logger.warning("Data exhausted at step %d; restarting iterator.", step)
            batch_iter = iter_token_batches(
                train_ds, tok, seq_len=args.seq_len, batch_size=args.batch_size,
                bos_id=bos_id, eos_id=eos_id, seed=args.seed + step,
            )
            batch = next(batch_iter)
        batch = jnp.asarray(batch)
        params, opt_state, loss = step_fn(params, opt_state, batch)
        loss_val = float(loss)
        losses.append(loss_val)

        if step % args.log_every == 0:
            avg = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
            elapsed = time.time() - t_start
            toks_per_s = step * args.batch_size * args.seq_len / elapsed
            logger.info("step %5d | loss %.4f (avg %.4f) | %.0f tok/s",
                        step, loss_val, avg, toks_per_s)
            if tracker is not None:
                tracker.log({"loss": loss_val, "loss_avg": avg,
                             "toks_per_s": toks_per_s}, step=step)

        if step % args.ckpt_every == 0 or step == args.steps:
            path = save_ckpt(args.out_dir, step, params, cfg)
            logger.info("Saved checkpoint: %s", path)


if __name__ == "__main__":
    main()
