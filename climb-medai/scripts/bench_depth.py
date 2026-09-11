#!/usr/bin/env python3
"""ColonStreamSfSNet timing microbenchmark -- no pipeline, just load + N
forward passes on the GPU path (the only one run.py uses).

Also doubles as the Dockerfile's build-time warm-up (small --n): it triggers
whatever Triton JIT-compiles the Mamba2 kernels on first use, so that doesn't
happen at --network=none run time.

Usage:
    conda activate sfsnet    # local dev
    python scripts/bench_depth.py --n 50
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(REPO_ROOT / "weights" / "colonstreamsfsnet" / "best_model.pth"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--n", type=int, default=50, help="timed forward passes")
    p.add_argument("--warmup", type=int, default=5, help="untimed forward passes first")
    return p.parse_args()


def main():
    args = parse_args()
    import torch

    from colonstreamsfsnet.infer import ColonStreamSfSNetDepth, invalid_pixel_mask

    print(f"Loading ColonStreamSfSNet on {args.device}...", flush=True)
    backend = ColonStreamSfSNetDepth(args.ckpt, args.device)

    rng = np.random.default_rng(0)
    frame_np = rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    frame = torch.from_numpy(frame_np).to(args.device)[..., [2, 1, 0]]
    invalid_mask = invalid_pixel_mask(frame)

    backend.reset_state()
    for _ in range(args.warmup):
        backend.estimate(frame, invalid_mask)

    times = []
    for _ in range(args.n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        backend.estimate(frame, invalid_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.asarray(times)
    print(f"n={args.n} at {args.width}x{args.height} (device={args.device})")
    print(
        f"  mean {times.mean() * 1000:.2f} ms  p50 {np.median(times) * 1000:.2f} ms  "
        f"p90 {np.percentile(times, 90) * 1000:.2f} ms  max {times.max() * 1000:.2f} ms"
    )
    print(
        f"  budget check: {times.mean():.4f} s/frame -- validity threshold is "
        "0.125 s/frame (whole pipeline, all components); no-penalty threshold is 0.025 s/frame"
    )


if __name__ == "__main__":
    main()
