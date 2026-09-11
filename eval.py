"""Frozen world-model benchmark. Edit train.py, not this file."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

CTX, HOR, N_WIN = 8, 16, 2048
FRAME_SHAPE, N_ACTIONS = (63, 63, 3), 17
EVAL_SEED, BATCH = 0, 64
CLIP, FEAT_SEED, THR = 4, 1234, 16 / 255
DATASET = "tinywmcraft_500k"
DATA_DIR = Path("data") / DATASET
DATASET_URL = f"https://huggingface.co/datasets/maharishiva/{DATASET}/resolve/main"


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("train.npz", "eval.npz"):
        destination = directory / name
        if destination.exists():
            print(f"Found {destination}", flush=True)
            continue
        temporary = destination.with_suffix(".npz.part")
        print(f"Downloading {name}", flush=True)
        urllib.request.urlretrieve(f"{DATASET_URL}/{name}", temporary)
        temporary.replace(destination)


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    frames, actions, dones = (data[key] for key in ("frames", "actions", "dones"))
    if actions.ndim != 2 or dones.shape != actions.shape:
        raise ValueError("actions and dones must have matching (time, env) shapes")
    t, e = actions.shape
    if frames.shape != (t + 1, e, *FRAME_SHAPE) or frames.dtype != np.uint8:
        raise ValueError(f"frames must be uint8 with shape {(t + 1, e, *FRAME_SHAPE)}")
    if not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions >= N_ACTIONS)):
        raise ValueError("Invalid actions")
    if not np.isin(dones, [0, 1]).all():
        raise ValueError("Invalid dones")
    return data


def sample_indices(data, n=N_WIN):
    """Keep the original benchmark's random draws and episode filtering."""
    dones = data["dones"].astype(bool)
    t, e = dones.shape
    span = CTX + HOR
    if n <= 0 or t <= span or e == 0:
        raise ValueError("Not enough data for evaluation windows")
    cumulative = np.concatenate([np.zeros((1, e), np.int64), np.cumsum(dones, axis=0)])
    starts = np.arange(1, t - span + 1)
    if not np.any(cumulative[starts + span - 1] == cumulative[starts]):
        raise ValueError("No evaluation windows without episode resets")
    rng = np.random.default_rng(EVAL_SEED)
    indices = []
    while len(indices) < n:
        env = int(rng.integers(0, e))
        start = int(rng.integers(1, t - span + 1))
        if cumulative[start + span - 1, env] == cumulative[start, env]:
            indices.append((start, env))
    return np.asarray(indices, dtype=np.int64)


def window_batch(data, indices):
    starts, envs = indices[:, 0], indices[:, 1]
    times = starts[:, None] + np.arange(CTX + HOR)[None]
    frames = data["frames"][times, envs[:, None]]
    actions = data["actions"][times - 1, envs[:, None]].astype(np.int32)
    actions[data["dones"][starts - 1, envs].astype(bool), 0] = 0
    return frames, actions


class FeatCNN(nnx.Module):
    def __init__(self, rngs):
        self.c1 = nnx.Conv(FRAME_SHAPE[-1] * CLIP, 32, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(32, 64, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(64, 128, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c4 = nnx.Conv(128, 32, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)

    def __call__(self, x):
        h = nnx.gelu(self.c1(x))
        h = nnx.gelu(self.c2(h))
        h = nnx.gelu(self.c3(h))
        return self.c4(h).reshape(x.shape[0], -1)


def feature_extractor():
    return nnx.jit(lambda x, m=FeatCNN(nnx.Rngs(FEAT_SEED)): m(x))


def clip_feats(feat, video):
    """Same clips as the original metric, with bounded temporary memory."""
    horizon = video.shape[1]
    batch = max(1, 256 // horizon)
    out = []
    for start in range(0, len(video), batch):
        v = video[start:start + batch].astype(np.float32) / 255
        pad = np.repeat(v[:, :1], CLIP - 1, 1)
        stacked = np.concatenate([pad, v], 1)
        clips = np.concatenate([stacked[:, t:t + horizon] for t in range(CLIP)], -1)
        flat = clips.reshape(-1, *FRAME_SHAPE[:2], FRAME_SHAPE[-1] * CLIP)
        out.append(np.asarray(feat(jnp.asarray(flat))))
    return np.concatenate(out)


def frechet(fr, fg):
    fr, fg = fr.astype(np.float64), fg.astype(np.float64)
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    cr = np.cov(fr, rowvar=False) + 1e-6 * np.eye(fr.shape[1])
    cg = np.cov(fg, rowvar=False) + 1e-6 * np.eye(fg.shape[1])
    eig = np.linalg.eigvals(cr @ cg)
    return float(((mu_r - mu_g) ** 2).sum() + np.trace(cr) + np.trace(cg)
                 - 2 * np.sum(np.sqrt(np.clip(eig.real, 0, None))))


def salient_mse(pred, real, last_ctx):
    real = real.astype(np.float32) / 255
    pred = pred.astype(np.float32) / 255
    sal = np.abs(real - last_ctx[:, None]).max(-1, keepdims=True) > THR
    return ((pred - real) ** 2 * sal).sum((1, 2, 3, 4)) / (sal.sum((1, 2, 3, 4)) + 1e-6)


def action_following(true_error, random_error):
    return float(np.mean((random_error > true_error + 1e-8)
                         + 0.5 * (np.abs(random_error - true_error) <= 1e-8)))


def make_gif(real, predicted, path, k=8, scale=4):
    import imageio.v2 as imageio

    k = min(k, len(real))
    separator = np.full((FRAME_SHAPE[0], 2, FRAME_SHAPE[-1]), (90, 90, 110), np.uint8)
    images = []
    for t in range(real.shape[1]):
        rows = [np.concatenate([real[i, t], separator, predicted[i, t]], 1) for i in range(k)]
        grid = np.concatenate(rows, 0)
        images.append(np.repeat(np.repeat(grid, scale, 0), scale, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images, duration=1000 / 6, loop=0)


def predict(model, frames, actions, future_actions, seed):
    import train

    predicted = np.asarray(train.rollout(
        model, frames[:, :CTX].copy(), actions[:, :CTX].copy(), future_actions.copy(), seed=seed,
    ))
    expected = (len(frames), future_actions.shape[1], *FRAME_SHAPE)
    if predicted.shape != expected or predicted.dtype != np.uint8:
        raise ValueError(f"rollout must return uint8 {expected}; got {predicted.dtype} {predicted.shape}")
    return predicted.copy()


def evaluate(model, data, indices):
    feat = feature_extractor()
    random_actions = np.random.default_rng(EVAL_SEED + 7).integers(
        0, N_ACTIONS, size=(len(indices), HOR)
    ).astype(np.int32)
    real_features, predicted_features, true_errors, random_errors = [], [], [], []
    preview = None
    for start in range(0, len(indices), BATCH):
        frames, actions = window_batch(data, indices[start:start + BATCH])
        batch_seed = int(np.random.SeedSequence([EVAL_SEED, start]).generate_state(1)[0])
        predicted = predict(model, frames, actions, actions[:, CTX:], batch_seed)
        randomized = predict(model, frames, actions, random_actions[start:start + BATCH], batch_seed)
        real = frames[:, CTX:]
        last_context = frames[:, CTX - 1].astype(np.float32) / 255
        real_features.append(clip_feats(feat, real))
        predicted_features.append(clip_feats(feat, predicted))
        true_errors.append(salient_mse(predicted, real, last_context))
        random_errors.append(salient_mse(randomized, real, last_context))
        if preview is None:
            preview = real[:8].copy(), predicted[:8].copy()
        if (start + BATCH) % 512 == 0:
            print(f"Evaluated {min(start + BATCH, len(indices))}/{len(indices)} windows", flush=True)
    scores = {
        "gFVD": frechet(np.concatenate(real_features), np.concatenate(predicted_features)),
        "AF": action_following(np.concatenate(true_errors), np.concatenate(random_errors)),
    }
    return scores, preview


def current_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the dataset and exit")
    parser.add_argument("--model", type=Path, help="Checkpoint; omit for the untrained baseline")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "eval.npz")
    parser.add_argument("--out", type=Path, default=Path("ledger.jsonl"))
    parser.add_argument("--tag")
    parser.add_argument("--note", default="")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(args.data.parent)
        return
    if args.train_seed < 0:
        parser.error("--train-seed must be nonnegative")

    import train

    model, metadata = train.load_model(args.model)
    train_s = float(metadata["train_s"])
    data = load_data(args.data)
    indices = sample_indices(data)
    commit = current_commit()
    start = time.perf_counter()
    scores, preview = evaluate(model, data, indices)
    tag = args.tag or (args.model.stem if args.model is not None else "persistence")
    note = args.note or ("Untrained copy-last-frame baseline" if args.model is None else "")
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": tag, "note": note, **scores,
        "train_seed": args.train_seed, "eval_seed": EVAL_SEED,
        "n_win": N_WIN, "ctx": CTX, "hor": HOR,
        "train_s": train_s, "eval_s": time.perf_counter() - start,
        "commit": commit, "checkpoint": str(args.model) if args.model is not None else None,
    }
    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as stream:
        stream.write(encoded + "\n")
    print(encoded, flush=True)
    if args.gif:
        path = args.out.parent / "gifs" / f"{tag}-train{args.train_seed}-eval{EVAL_SEED}.gif"
        make_gif(*preview, path)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
