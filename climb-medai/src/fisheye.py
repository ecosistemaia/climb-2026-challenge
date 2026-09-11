"""Kannala-Brandt fisheye handling for the pose-only front-end.

Only *point* coordinates are rectified -- never the image or depth map, since
warping those would lose peripheral pixels and push the RGB/depth networks
out of their training distribution.

Undistortion is a manual Newton solve rather than `cv2.fisheye.undistortPoints`
because it needs theta directly, runs at fixed cost (no silent non-convergence),
and stays on the GPU.
"""

import math

import numpy as np
import torch

# Above this incidence angle tan(theta) blows up and rectified coordinates
# become meaningless, not just large.
DEFAULT_THETA_MAX_DEG = 50.0

_NEWTON_ITERS = 10


def kb_theta_from_theta_d(theta_d, k1, k2, k3, k4, iters=_NEWTON_ITERS):
    """Invert theta_d = theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8) via
    Newton's method, seeded at theta = theta_d."""
    theta = theta_d.clone()
    for _ in range(iters):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        f = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d
        df = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8
        theta = theta - f / torch.clamp(df, min=1e-8)  # clamp guards a pathological calibration, not real lenses
    return theta


def undistort_points_kb(x, y, K, D, theta_max_deg=DEFAULT_THETA_MAX_DEG):
    """Rectify fisheye pixel coordinates to the nominal pinhole camera.

    `x, y`: 1-D tensors of distorted pixel coordinates. `K`=(fx,fy,cx,cy) is
    reused as the output projection, so rectified coordinates stay consistent
    with the intrinsics the lifting step uses. `D`=(k1,k2,k3,k4).

    Returns `(x_u, y_u, valid)`; rejected entries still carry a finite value
    (never NaN) but must be filtered by `valid`.
    """
    fx, fy, cx, cy = (float(v) for v in K)
    k1, k2, k3, k4 = (float(v) for v in D)

    x_d = (x - cx) / fx
    y_d = (y - cy) / fy
    theta_d = torch.sqrt(x_d * x_d + y_d * y_d)

    theta = kb_theta_from_theta_d(theta_d, k1, k2, k3, k4)

    theta_max = math.radians(float(theta_max_deg))
    valid = torch.isfinite(theta) & (theta >= 0.0) & (theta <= theta_max)

    # tan(theta)/theta_d, with the theta_d -> 0 limit (=1) handled explicitly
    # (0/0 at the optical centre otherwise).
    theta_safe = torch.where(valid, theta, torch.zeros_like(theta))
    scale = torch.where(
        theta_d > 1e-8,
        torch.tan(theta_safe) / torch.clamp(theta_d, min=1e-8),
        torch.ones_like(theta_d),
    )
    scale = torch.where(valid, scale, torch.ones_like(scale))

    x_u = x_d * scale * fx + cx
    y_u = y_d * scale * fy + cy
    valid &= torch.isfinite(x_u) & torch.isfinite(y_u)
    return x_u, y_u, valid


def distort_points_kb(x_u, y_u, K, D):
    """Forward KB projection -- the inverse of `undistort_points_kb`. Used by
    the round-trip test only, not on the runtime path."""
    fx, fy, cx, cy = (float(v) for v in K)
    k1, k2, k3, k4 = (float(v) for v in D)

    a = (x_u - cx) / fx
    b = (y_u - cy) / fy
    r = torch.sqrt(a * a + b * b)
    theta = torch.atan(r)
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    theta_d = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)
    scale = torch.where(r > 1e-8, theta_d / torch.clamp(r, min=1e-8), torch.ones_like(r))
    return a * scale * fx + cx, b * scale * fy + cy


def rectified_radius_for_theta(theta_deg, fx):
    """Rectified radius in pixels for a given incidence angle: fx * tan(theta).
    Handy for choosing `theta_max_deg` against an image size."""
    return float(fx) * math.tan(math.radians(float(theta_deg)))


def kb_params_from_cfg(cam_cfg):
    """Read the fisheye block out of a sequence config.

    Returns `(enabled, K, D, theta_max_deg)`; `enabled` is False (caller
    leaves coordinates untouched) unless `undistort_points` is set and four
    distortion coefficients are supplied.
    """
    enabled = bool(cam_cfg.get('undistort_points', False))
    if not enabled:
        return False, None, None, None

    model = str(cam_cfg.get('distortion_model', 'kannala_brandt')).lower()
    if model not in ('kannala_brandt', 'kb', 'kannalabrandt8', 'opencv_fisheye', 'fisheye'):
        raise ValueError(
            f"cam.undistort_points is set but cam.distortion_model={model!r} is not a "
            "Kannala-Brandt / OPENCV_FISHEYE model. Refusing to guess a distortion model."
        )

    coeffs = cam_cfg.get('distortion', None)
    if coeffs is None or len(coeffs) != 4:
        raise ValueError(
            "cam.undistort_points requires cam.distortion with exactly 4 Kannala-Brandt "
            f"coefficients (k1..k4); got {coeffs!r}."
        )

    K = (
        float(cam_cfg['fx']),
        float(cam_cfg['fy']),
        float(cam_cfg['cx']),
        float(cam_cfg['cy']),
    )
    D = tuple(float(c) for c in coeffs)
    theta_max_deg = float(cam_cfg.get('theta_max_deg', DEFAULT_THETA_MAX_DEG))
    if not (0.0 < theta_max_deg < 90.0):
        raise ValueError(f"cam.theta_max_deg must be in (0, 90); got {theta_max_deg}.")
    return True, K, D, theta_max_deg


def undistort_points_kb_numpy(xy, K, D, theta_max_deg=DEFAULT_THETA_MAX_DEG):
    """numpy convenience wrapper over `undistort_points_kb` for (N, 2) arrays."""
    xy = np.asarray(xy, dtype=np.float64)
    x = torch.from_numpy(xy[:, 0])
    y = torch.from_numpy(xy[:, 1])
    x_u, y_u, valid = undistort_points_kb(x, y, K, D, theta_max_deg)
    return np.stack([x_u.numpy(), y_u.numpy()], axis=-1), valid.numpy()
