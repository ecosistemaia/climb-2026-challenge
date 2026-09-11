#!/usr/bin/env python3
"""climb-medai: CLiMB 2026 pose-only submission entrypoint.

Input:
  `/input/*.mp4` -- a flat directory of endoscopy videos.

Process, per video:
  1. Decode frames (`src/video_reader.py`).
  2. Estimate monocular metric depth (`colonstreamsfsnet/infer.py`).
  3. Estimate dense optical flow (RAFT).
  4. Rectify (fisheye) and solve frame-to-frame pose via PnP (`src/vo.py`).
  5. Chain the relative poses into a camera-to-world trajectory.

Output:
  `--num-runs` independent runs per video, each with `camera_trajectory/`,
  `3D_maps/`, and `runtime.txt` -- the full CLiMB submission tree.

Notes:
  - Weights are loaded once for the whole process, not per run.
  - `init_seconds` is excluded from scoring; runs stay independent because
    RANSAC sampling and floating-point nondeterminism vary run to run even
    with warm weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from colonstreamsfsnet.infer import ColonStreamSfSNetDepth, invalid_pixel_mask
from src.video_reader import VideoFrameExtractor
from src.vo import VOFrontend


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# The method's operating point: every VO/PnP/depth/submap hyperparameter and
# the camera calibration, baked in -- not overridable from the CLI. The
# submission image never sets CLIMB_MEDAI_CONFIG, so this always resolves to
# v11 there; the env var exists only so local experiment batteries can point
# at a sibling config (e.g. configs/climb_v9c.yaml) without editing this file.
CONFIG_PATH = Path(os.environ.get("CLIMB_MEDAI_CONFIG", str(REPO_ROOT / "configs" / "climb_v11.yaml")))


def load_config(path: Path = CONFIG_PATH) -> dict:
    sections = yaml.safe_load(path.read_text())
    return {k: v for section in sections.values() for k, v in section.items()}


TRAJ_HEADER = "# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n"
CONFIG = load_config()
KB_D = (CONFIG["k1"], CONFIG["k2"],
        CONFIG["k3"], CONFIG["k4"])
W, H = CONFIG["work_width"], CONFIG["work_height"]
FX, FY = CONFIG["work_fx"], CONFIG["work_fy"]
CX, CY = CONFIG["work_cx"], CONFIG["work_cy"]
BLACK_THRESH = 10  # endoscope vignette cutoff, not exposed as a tuning knob
SPECULAR_THRESH = None if CONFIG["specular_thresh"] < 0 else CONFIG["specular_thresh"]
MASK_DILATE_PX = CONFIG["mask_dilate_px"]


def parse_args():
    """I/O and run control only. The method's hyperparameters are not CLI
    flags -- they live in CONFIG_PATH (see load_config)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default="/input")
    parser.add_argument("output", nargs="?", default="/output")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def find_videos(input_dir: Path, extensions=(".mp4",)):
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def load_raft(device):
    model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
    return model.eval()


def load_colonstreamsfsnet(ckpt_path, device, resolution=(H, W)):
    return ColonStreamSfSNetDepth(ckpt_path=ckpt_path, device=device, resolution=tuple(resolution))


def write_points3d(path: Path):
    """Dummy fixed point cloud -- points3D.txt isn't scored, it just can't be
    empty or the evaluator crashes. See submission_instructions/README.md.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR\n")
        f.write("1 0.000000 0.000000 0.050000 255 0 0 0.0\n")
        f.write("2 0.010000 0.000000 0.050000 0 255 0 0.0\n")
        f.write("3 0.000000 0.010000 0.050000 0 0 255 0.0\n")
        f.write("4 0.010000 0.010000 0.050000 255 255 255 0.0\n")


def _traj_row(frame_id: int, pose_c2w: np.ndarray, fps: float) -> str:
    timestamp = (int(frame_id) - 1) / fps
    t = pose_c2w[:3, 3]
    qx, qy, qz, qw = Rotation.from_matrix(pose_c2w[:3, :3]).as_quat()  # scipy: xyzw
    return (f"{timestamp:.6f},{int(frame_id):06d}.png,"
            f"{t[0]:.9f},{t[1]:.9f},{t[2]:.9f},"
            f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n")               # challenge order: qw first


def write_traj_file(path: Path, rows, fps: float):
    """rows = [(frame_id, pose_c2w 4x4), ...]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(TRAJ_HEADER)
        for fid, pose in rows:
            f.write(_traj_row(fid, pose, fps))


def segment_trajectory(records, cut_n: int, reopen_m: int):
    """Cut the trajectory into separate sub-maps wherever tracking was lost
    for a while, instead of dragging one long stretch of frozen/guessed poses
    into the output.

    `records` = [(frame_id, kind, T21_np|None), ...] in frame order:
    'anchor' = frame 1 (no VO yet), 'solved' = VO gave a real pose, 'hold' =
    VO failed this frame (pose was just repeated).

    Rule: if `cut_n` 'hold's happen in a row, that's a real tracking loss --
    close the current sub-map right there (those `cut_n` frames are dropped,
    not written anywhere) and wait. Once `reopen_m` 'solved' frames happen in
    a row, trust that tracking recovered and start a brand-new sub-map from
    identity pose.

    Returns a list of sub-maps, each a list of (frame_id, pose_c2w 4x4) --
    every frame appears in at most one sub-map, in time order.
    """
    I4 = np.eye(4)
    segments, cur = [], []
    pose = I4.copy()
    state = "NORMAL"
    pending_hold = []          # frame ids of a possible episode (NORMAL)
    wait_buf = []              # (fid, T21) during WAITING

    def flush_pending():
        nonlocal pending_hold
        for h in pending_hold:
            cur.append((h, pose.copy()))
        pending_hold = []

    for fid, kind, T21 in records:
        if state == "NORMAL":
            if kind == "anchor":
                flush_pending()
                cur.append((fid, pose.copy()))
            elif kind == "solved":
                flush_pending()
                pose = pose @ np.linalg.inv(T21)
                cur.append((fid, pose.copy()))
            else:  # hold
                pending_hold.append(fid)
                if len(pending_hold) >= cut_n:
                    if cur:
                        segments.append(cur)
                    cur = []
                    pending_hold = []
                    pose = I4.copy()
                    state = "WAITING"
                    wait_buf = []
        else:  # WAITING
            if kind == "anchor":
                continue
            if kind == "solved":
                wait_buf.append((fid, T21))
                if len(wait_buf) >= reopen_m:
                    p = I4.copy()
                    cur = [(wait_buf[0][0], p.copy())]
                    for bfid, bT in wait_buf[1:]:
                        p = p @ np.linalg.inv(bT)
                        cur.append((bfid, p.copy()))
                    pose = p
                    wait_buf = []
                    state = "NORMAL"
            else:  # hold -> reset the recovery streak
                wait_buf = []

    if state == "NORMAL":
        flush_pending()
        if cur:
            segments.append(cur)
    return segments


def process_run(video_path: Path, sequence_root: Path, sfsnet, raft, vo_cfg, device,
                submap_cut_n: int = 0, submap_reopen_m: int = 10):
    map_root = sequence_root / "3D_maps" / "0"
    trajectory_root = sequence_root / "camera_trajectory"

    # Init: everything before the per-frame loop.
    t_init_start = time.perf_counter()
    sfsnet.reset_state()
    vo = VOFrontend(raft, vo_cfg)
    emit_submaps = submap_cut_n > 0
    traj_file = None
    
    if not emit_submaps:
        trajectory_path = trajectory_root / "cam_traj_map_000.txt"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        traj_file = trajectory_path.open("w", encoding="utf-8")
        traj_file.write(TRAJ_HEADER)

    extractor = VideoFrameExtractor(video_path, start_index=1)
    fps = extractor.fps if extractor.fps and extractor.fps > 0 else 40.0

    # Processing: the per-frame loop.
    t_processing_start = time.perf_counter()

    pose_c2w = np.eye(4, dtype=np.float64)
    prev_color = None
    prev_depth = None
    n_frames = 0
    
    # Per-run diagnostics
    diagnostics = {
        "vo_attempted_frames": 0,
        "vo_clamp_trans_frames": 0,
        "vo_clamp_rot_frames": 0,
        "vo_fail": {
            'few_points': 0,
            'few_points_after_undistort': 0,
            'few_points_for_pnp': 0,
            'pnp_exception': 0,
            'pnp_failed': 0,
            'few_pnp_inliers': 0,
            'pnp_projection_failed': 0,
            'pnp_non_finite': 0,
            'non_finite': 0,
            'solver_failed': 0,
        },
    }
    submap_records = [] if emit_submaps else None  # (frame_id, kind, T21) per frame

    for frame_id, bgr, _meta in extractor:
        bgr_work = cv2.resize(bgr, (W, H))

        # BGR->RGB on-device
        frame = torch.from_numpy(bgr_work).to(device)[..., [2, 1, 0]]

        # Black-border + specular mask, computed once.
        invalid_mask = invalid_pixel_mask(
            frame,
            black_thresh=BLACK_THRESH,
            specular_thresh=SPECULAR_THRESH,
            dilate_px=MASK_DILATE_PX,
        )

        # Monocular Depth Estimation
        depth_t_mm = sfsnet.estimate(frame, invalid_mask)

        # VOFrontend.estimate_relative_pose expects.
        curr_color = frame.unsqueeze(0).float() / 255.0
        curr_depth = depth_t_mm.unsqueeze(0)  # mm

        # We have a previous frame to compare against, so run VO.
        if prev_color is not None:
            T21, _info = vo.estimate_relative_pose(prev_color, curr_color, prev_depth, curr_depth)
            diagnostics["vo_attempted_frames"] += 1
            reason = _info.get('vo_reason') if _info is not None else None
            
            if reason in diagnostics["vo_fail"]:
                diagnostics["vo_fail"][reason] += 1
            elif reason == 'ok' and _info is not None:
                if _info.get('vo_clamp_trans'):
                    diagnostics["vo_clamp_trans_frames"] += 1
                if _info.get('vo_clamp_rot'):
                    diagnostics["vo_clamp_rot_frames"] += 1
            T21_np = None
            if T21 is not None:
                T21_np = T21.detach().cpu().numpy().astype(np.float64)
                pose_c2w = pose_c2w @ np.linalg.inv(T21_np)
            # else: VO produced nothing usable this frame -- hold the previous
            # pose rather than emit a wrong one. Frame is still reported (RTF
            # wants every frame to get *a* pose), just not updated.

            if submap_records is not None:
                submap_records.append((frame_id, "solved", T21_np) if T21_np is not None
                                      else (frame_id, "hold", None))
        elif submap_records is not None:
            submap_records.append((frame_id, "anchor", None))

        if traj_file is not None:
            traj_file.write(_traj_row(frame_id, pose_c2w, fps))

        prev_color, prev_depth = curr_color, curr_depth
        n_frames += 1

    t_processing_end = time.perf_counter()

    # Post-processing (shutdown, map serialization)
    if traj_file is not None:
        traj_file.close()

    submap_sizes = [n_frames]
    if emit_submaps:
        segments = segment_trajectory(submap_records, submap_cut_n, submap_reopen_m)
        if not segments:
            anchor_fid = submap_records[0][0] if submap_records else 1
            segments = [[(anchor_fid, np.eye(4))]]
        submap_sizes = [len(s) for s in segments]
        for i, seg in enumerate(segments):
            write_traj_file(trajectory_root / f"cam_traj_map_{i:03d}.txt", seg, fps)
            write_points3d(sequence_root / "3D_maps" / str(i) / "points3D.txt")
    else:
        write_points3d(map_root / "points3D.txt")

    # runtime.txt: the two contract keys the evaluator parses. init_seconds =
    # everything before the frame loop; processing_seconds = the loop itself
    # (the only scored quantity).
    (sequence_root / "runtime.txt").write_text(
        f"init_seconds={t_processing_start - t_init_start:.6f}\n"
        f"processing_seconds={t_processing_end - t_processing_start:.6f}\n"
    )

    # diagnostics.json: sidecar to runtime.txt, evaluator ignores it.
    diagnostics.update({
        "n_frames": n_frames,
        "init_seconds": t_processing_start - t_init_start,
        "processing_seconds": t_processing_end - t_processing_start,
        "n_maps": len(submap_sizes),
        "submap_sizes": submap_sizes,
        "submap_cut_n": submap_cut_n,
        "submap_reopen_m": submap_reopen_m if emit_submaps else None,
    })
    (sequence_root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return n_frames


def process_sequence(video_path: Path, output_dir: Path, num_runs: int, device,
                      sfsnet, raft, vo_cfg,
                      submap_cut_n: int = 0, submap_reopen_m: int = 10):
    sequence = video_path.stem

    seq_t0 = time.perf_counter()
    for run_id in range(1, num_runs + 1):
        sequence_root = output_dir / sequence / str(run_id)
        n_frames = process_run(video_path, sequence_root, sfsnet, raft, vo_cfg, device,
                               submap_cut_n=submap_cut_n, submap_reopen_m=submap_reopen_m)
        print(f"    run {run_id}/{num_runs}: frames={n_frames}", flush=True)
    print(f"  {sequence}: runs={num_runs} runtime={time.perf_counter() - seq_t0:.2f}s", flush=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    config = {**vars(args), **CONFIG}
    print(f"Config (CLI + {CONFIG_PATH.name}):", flush=True)
    for key, value in config.items():
        print(f"  {key}: {value}", flush=True)
    print("-" * 40, flush=True)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    # Find all .mp4 files in the input directory.
    videos = find_videos(input_dir)
    if not videos:
        raise SystemExit(f"No .mp4 files found in {input_dir}")

    print(f"Found {len(videos)} video(s):", flush=True)
    for video_path in videos:
        print(f"  {video_path.name}", flush=True)
    print("-" * 40, flush=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU.", flush=True)

    # Load the RAFT optical flow model and the ColonStreamSfSNet depth model once for all runs.
    print(f"Loading RAFT (small) on {device}...", flush=True)
    raft = load_raft(device)

    print(f"Loading ColonStreamSfSNet on {device}...", flush=True)
    depth_ckpt = REPO_ROOT / config["depth_ckpt"]
    sfsnet = load_colonstreamsfsnet(depth_ckpt, device)

    print("Warming up ColonStreamSfSNet (Triton/Mamba2 kernel compile)...", flush=True)
    t_warmup_start = time.perf_counter()
    dummy_frame = torch.from_numpy(np.zeros((H, W, 3), dtype=np.uint8)).to(device)
    sfsnet.reset_state()
    sfsnet.estimate(dummy_frame)  # compiles seqlen_offset==0 (prefill) path
    sfsnet.estimate(dummy_frame)  # compiles seqlen_offset>0 (inference) path
    print(f"Warmup done in {time.perf_counter() - t_warmup_start:.2f}s", flush=True)

    # VOFrontend configuration
    vo_cfg = SimpleNamespace(**config, fisheye_K=(FX, FY, CX, CY), fisheye_D=KB_D, H=H, W=W)

    # Process each video in the input directory, running the specified number of independent runs per video.
    t0 = time.perf_counter()
    for video_path in videos:
        print(f"== {video_path.name} ==", flush=True)
        process_sequence(video_path,
                         output_dir,
                         args.num_runs,
                         device,
                         sfsnet,
                         raft,
                         vo_cfg,
                         submap_cut_n=config["submap_cut_n"],
                         submap_reopen_m=config["submap_reopen_m"])
    print(f"Total: {time.perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
