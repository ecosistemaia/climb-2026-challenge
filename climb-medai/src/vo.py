"""Pose-only VO front-end.

Per frame pair:
  1. RAFT dense optical flow -> sparse correspondences.
  2. Fisheye point rectification.
  3. Lift to 3D.
  4. PnP pose solve.

Notes:
  - Single correspondence source (RAFT), single solver (PnP).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from .fisheye import undistort_points_kb


# Rectify (if configured) and lift 2D correspondences to 3D


def lift_correspondences(x1_sel, y1_sel, d1_sel, x2_sel, y2_sel, fx, fy, cx, cy, cfg,
                          endpoint1_precomputed=None):
    """Rectify tracked points (if configured) and back-project them to 3D.

    `endpoint1_precomputed`, if given, is a precomputed (x1_u, y1_u, valid1)
    lookup for the first endpoint (see `VOFrontend._rectify_lut`); the second
    endpoint is sub-pixel (RAFT-flow-shifted), so it's always rectified live.
    Only 2D coords are rectified -- depth was sampled at the original
    distorted pixels and is never warped. `d` is Z along the optical axis
    (ColonStreamSfSNet's convention), so this is a plain pinhole lift.

    Returns `(X1, x1, y1, x2, y2, info)`; `X1` is `None` if too few points survive.
    """
    info = {'vo_undistort_points': bool(cfg.undistort_points_enabled)}

    if cfg.undistort_points_enabled:
        n_before = int(x1_sel.shape[0])
        if endpoint1_precomputed is not None:
            x1_u, y1_u, valid1 = endpoint1_precomputed
        else:
            x1_u, y1_u, valid1 = undistort_points_kb(x1_sel, y1_sel, cfg.fisheye_K, cfg.fisheye_D, cfg.fisheye_theta_max_deg)
        x2_u, y2_u, valid2 = undistort_points_kb(x2_sel, y2_sel, cfg.fisheye_K, cfg.fisheye_D, cfg.fisheye_theta_max_deg)
        # A correspondence is only usable if BOTH endpoints rectify: a point
        # that leaves the theta cap between frames would otherwise pair a
        # real observation with a saturated one.
        keep = valid1 & valid2
        n_after = int(keep.sum().item())
        info.update(
            {
                'vo_undistort_points_in': n_before,
                'vo_undistort_points_out': n_after,
                'vo_undistort_theta_max_deg': float(cfg.fisheye_theta_max_deg),
            }
        )
        if n_after < cfg.min_points:
            return None, None, None, None, None, info
        if n_after < n_before:
            x1_sel, y1_sel, d1_sel = x1_u[keep], y1_u[keep], d1_sel[keep]
            x2_sel, y2_sel = x2_u[keep], y2_u[keep]
        else:
            x1_sel, y1_sel = x1_u, y1_u
            x2_sel, y2_sel = x2_u, y2_u

    X1 = torch.stack([(x1_sel - cx) / fx * d1_sel, (y1_sel - cy) / fy * d1_sel, d1_sel], dim=-1)
    return X1, x1_sel, y1_sel, x2_sel, y2_sel, info


# PnP solver


def estimate_vo_transform_pnp(X1: torch.Tensor, x2_sel: torch.Tensor, y2_sel: torch.Tensor,
                               fx: float, fy: float, cx: float, cy: float, cfg):
    """solvePnPRansac + optional refine. Adapted from Track2Map's
    `SceneOptimizer._estimate_vo_transform_pnp`."""
    n_points = int(X1.shape[0])
    min_inliers = max(32, int(cfg.min_inliers))
    if n_points < max(6, min_inliers):
        return None, None, {'vo_used': False, 'vo_reason': 'few_points_for_pnp', 'vo_inlier_points': 0}

    obj_pts = X1.detach().cpu().numpy().astype(np.float32)
    img_pts = torch.stack([x2_sel, y2_sel], dim=-1).detach().cpu().numpy().astype(np.float32)
    K = np.array([[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]], dtype=np.float64)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=K,
            distCoeffs=None,
            reprojectionError=float(cfg.pnp_reproj_err),
            confidence=float(np.clip(cfg.pnp_confidence, 0.5, 0.9999)),
            iterationsCount=max(1, int(cfg.pnp_iterations)),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None, None, {'vo_used': False, 'vo_reason': 'pnp_exception', 'vo_inlier_points': 0}

    if (not ok) or rvec is None or tvec is None:
        return None, None, {'vo_used': False, 'vo_reason': 'pnp_failed', 'vo_inlier_points': 0}

    inlier_idx = np.arange(n_points, dtype=np.int32) if inliers is None else inliers.reshape(-1).astype(np.int32)
    if inlier_idx.size < min_inliers:
        return None, None, {
            'vo_used': False, 'vo_reason': 'few_pnp_inliers', 'vo_inlier_points': int(inlier_idx.size),
        }

    if cfg.pnp_refine and inlier_idx.size >= 6:
        obj_in = obj_pts[inlier_idx]
        img_in = img_pts[inlier_idx]
        try:
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, K, None, rvec, tvec)
            else:
                cv2.solvePnP(obj_in, img_in, K, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            pass

    try:
        R_np, _ = cv2.Rodrigues(rvec)
    except cv2.error:
        return None, None, {
            'vo_used': False, 'vo_reason': 'pnp_projection_failed', 'vo_inlier_points': int(inlier_idx.size),
        }

    # Computed here in plain numpy off cv2's own output -- no GPU round trip.
    trace_np = float(np.clip((np.trace(R_np) - 1.0) * 0.5, -1.0, 1.0))
    rot_deg_raw = float(np.degrees(np.arccos(trace_np)))
    trans_norm_raw = float(np.linalg.norm(tvec))

    R = torch.from_numpy(R_np).to(device=X1.device, dtype=X1.dtype)
    t = torch.from_numpy(tvec.reshape(3)).to(device=X1.device, dtype=X1.dtype)
    if (not torch.isfinite(R).all()) or (not torch.isfinite(t).all()):
        return None, None, {
            'vo_used': False, 'vo_reason': 'pnp_non_finite', 'vo_inlier_points': int(inlier_idx.size),
        }
    return R, t, {
        'vo_used': True,
        'vo_reason': 'ok',
        'vo_inlier_points': int(inlier_idx.size),
        'vo_rot_deg_raw': rot_deg_raw,
        'vo_trans_norm_raw': trans_norm_raw,
    }


def apply_pose_clamps(R: torch.Tensor, t: torch.Tensor, trans_norm_raw: float,
                       rot_deg_raw: float, cfg):
    """Clamp a PnP relative pose to cfg.max_trans / cfg.max_rot_deg -- scales
    magnitude down, direction unchanged, never rejects a frame.

    Returns (R, t, trans_norm, rot_deg, clamp_trans_fired, clamp_rot_fired).
    """
    trans_norm = float(trans_norm_raw)
    rot_deg = float(rot_deg_raw)
    clamp_trans_fired = False
    clamp_rot_fired = False
    if cfg.max_trans > 0.0 and trans_norm > cfg.max_trans:
        t = t * (cfg.max_trans / max(trans_norm, 1e-12))
        trans_norm = float(cfg.max_trans)
        clamp_trans_fired = True
    if cfg.max_rot_deg > 0.0 and rot_deg > cfg.max_rot_deg:
        rot = Rotation.from_matrix(R.detach().cpu().numpy())
        rotvec = rot.as_rotvec()
        rotvec = rotvec * (cfg.max_rot_deg / max(rot_deg, 1e-12))
        R = torch.from_numpy(Rotation.from_rotvec(rotvec).as_matrix()).to(device=R.device, dtype=R.dtype)
        rot_deg = float(cfg.max_rot_deg)
        clamp_rot_fired = True
    return R, t, trans_norm, rot_deg, clamp_trans_fired, clamp_rot_fired


# Orchestrator


class VOFrontend:
    """RAFT correspondences -> rectify -> lift -> PnP.

    One instance per running process (run.py builds a fresh one per run). The
    front-end is stateless across frames -- every call to
    `estimate_relative_pose` is independent.
    """

    def __init__(self, raft_model, cfg):
        self.raft = raft_model
        self.cfg = cfg
        self._mesh_cache_key = None
        self._mesh_xx = None
        self._mesh_yy = None
        self._rectify_lut_cache_key = None
        self._rectify_lut_x = None
        self._rectify_lut_y = None
        self._rectify_lut_valid = None

    def _pixel_grid(self, H: int, W: int, device, dtype):
        """Cached (1, H, W) pixel-coordinate grids, keyed on (H, W, device, dtype)."""
        key = (H, W, device, dtype)
        if self._mesh_cache_key != key:
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing='ij',
            )
            self._mesh_xx = xx.unsqueeze(0)
            self._mesh_yy = yy.unsqueeze(0)
            self._mesh_cache_key = key
        return self._mesh_xx, self._mesh_yy

    def _rectify_lut(self, H: int, W: int, device, dtype):
        """Cached (H, W) fisheye-rectification LUT (x_u, y_u, valid) for the
        first correspondence endpoint -- exact, just evaluated once per shape."""
        key = (H, W, device, dtype)
        if self._rectify_lut_cache_key != key:
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing='ij',
            )
            x_u, y_u, valid = undistort_points_kb(
                xx.reshape(-1), yy.reshape(-1),
                self.cfg.fisheye_K, self.cfg.fisheye_D, self.cfg.fisheye_theta_max_deg,
            )
            self._rectify_lut_x = x_u.reshape(H, W)
            self._rectify_lut_y = y_u.reshape(H, W)
            self._rectify_lut_valid = valid.reshape(H, W)
            self._rectify_lut_cache_key = key
        return self._rectify_lut_x, self._rectify_lut_y, self._rectify_lut_valid

    def estimate_relative_pose(self, prev_color: torch.Tensor, curr_color: torch.Tensor,
                                prev_depth: torch.Tensor, curr_depth: torch.Tensor):
        """Returns `(T21, info)`. `T21` is a 4x4 torch tensor (frame-1 -> frame-2
        rigid transform) or `None` if VO could not produce a pose this frame
        (caller should fall back to holding the previous pose)."""
        cfg = self.cfg
        if prev_color is None or curr_color is None or prev_depth is None or curr_depth is None:
            return None, None

        H, W = cfg.H, cfg.W
        fx, fy, cx, cy = cfg.fisheye_K
        d1 = prev_depth.to(device=curr_color.device, dtype=curr_color.dtype)
        d2 = curr_depth.to(device=curr_color.device, dtype=curr_color.dtype)

        # Normalize to [-1, 1] for RAFT, permute to (B, C, H, W) for RAFT.
        raft_prev = 2 * prev_color.permute(0, 3, 1, 2) - 1.0
        raft_curr = 2 * curr_color.permute(0, 3, 1, 2) - 1.0
        with torch.no_grad():
            flow = self.raft(raft_prev, raft_curr, num_flow_updates=cfg.num_flow_updates)[-1]
        flow_x = flow[:, 0]
        flow_y = flow[:, 1]
        flow_mag = torch.sqrt(flow_x * flow_x + flow_y * flow_y)

        def _correspond_and_solve():
            # Sample the flow field at every pixel.
            xx, yy = self._pixel_grid(H, W, flow.device, flow.dtype)
            x2 = xx + flow_x
            y2 = yy + flow_y

            valid = (x2 >= 0.0) & (x2 <= (W - 1)) & (y2 >= 0.0) & (y2 <= (H - 1))
            if cfg.max_flow > 0.0:
                valid &= flow_mag <= cfg.max_flow

            grid_x = (2.0 * x2 / max(W - 1, 1)) - 1.0
            grid_y = (2.0 * y2 / max(H - 1, 1)) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)
            d2_warp = F.grid_sample(d2.unsqueeze(1), grid, mode='bilinear', align_corners=True).squeeze(1)

            valid &= torch.isfinite(d1) & torch.isfinite(d2_warp)
            valid &= (d1 > cfg.min_depth) & (d2_warp > cfg.min_depth)
            if cfg.max_depth > cfg.min_depth:
                valid &= (d1 < cfg.max_depth) & (d2_warp < cfg.max_depth)

            valid_idx = torch.where(valid.squeeze(0))
            n_valid = int(valid_idx[0].numel())
            if n_valid < cfg.min_points:
                return None, {'vo_valid_points': n_valid, 'vo_used': False, 'vo_reason': 'few_points'}

            if cfg.max_points > 0 and n_valid > cfg.max_points:
                # randint (with replacement)
                perm = torch.randint(0, n_valid, (cfg.max_points,), device=flow.device)
                ys = valid_idx[0][perm]
                xs = valid_idx[1][perm]
            else:
                ys = valid_idx[0]
                xs = valid_idx[1]

            x1_sel = xs.float()
            y1_sel = ys.float()
            x2_sel = x2.squeeze(0)[ys, xs]
            y2_sel = y2.squeeze(0)[ys, xs]
            d1_sel = d1.squeeze(0)[ys, xs]

            endpoint1_precomputed = None
            if cfg.undistort_points_enabled:
                lut_x, lut_y, lut_valid = self._rectify_lut(H, W, flow.device, flow.dtype)
                endpoint1_precomputed = (lut_x[ys, xs], lut_y[ys, xs], lut_valid[ys, xs])

            X1, x1_sel, y1_sel, x2_sel, y2_sel, undistort_info = lift_correspondences(
                x1_sel, y1_sel, d1_sel, x2_sel, y2_sel, fx, fy, cx, cy, cfg,
                endpoint1_precomputed=endpoint1_precomputed,
            )
            if X1 is None:
                info = {'vo_used': False, 'vo_reason': 'few_points_after_undistort'}
                info.update(undistort_info)
                return None, info

            R, t, solver_info = estimate_vo_transform_pnp(X1, x2_sel, y2_sel, fx, fy, cx, cy, cfg)

            if R is None or t is None:
                info = {
                    'vo_valid_points': int(X1.shape[0]),
                    'vo_used': False,
                    'vo_reason': 'solver_failed',
                    'vo_inlier_points': 0,
                }
                if solver_info is not None:
                    info.update(solver_info)
                info.update(undistort_info)
                return None, info

            rot_deg_raw = float((solver_info or {}).get('vo_rot_deg_raw', float('nan')))
            trans_norm_raw = float((solver_info or {}).get('vo_trans_norm_raw', float('nan')))
            if not torch.isfinite(R).all() or not torch.isfinite(t).all():
                info = {'vo_valid_points': int(X1.shape[0]), 'vo_used': False, 'vo_reason': 'non_finite'}
                info.update(undistort_info)
                return None, info

            # Apply the configured clamps to the PnP output, if any.
            R, t, trans_norm, rot_deg, clamp_trans_fired, clamp_rot_fired = apply_pose_clamps(
                R, t, trans_norm_raw, rot_deg_raw, cfg,
            )

            T21 = torch.eye(4, device=R.device, dtype=R.dtype)
            T21[:3, :3] = R
            T21[:3, 3] = t
            info = {
                'vo_valid_points': int(X1.shape[0]),
                'vo_inlier_points': int((solver_info or {}).get('vo_inlier_points', X1.shape[0])),
                'vo_used': True,
                'vo_reason': 'ok',
                'vo_trans_norm': trans_norm,
                'vo_rot_deg': rot_deg,
                'vo_trans_norm_raw': float(trans_norm_raw),
                'vo_rot_deg_raw': float(rot_deg_raw),
                'vo_clamp_trans': clamp_trans_fired,
                'vo_clamp_rot': clamp_rot_fired,
            }
            info.update(undistort_info)
            return T21, info

        return _correspond_and_solve()
