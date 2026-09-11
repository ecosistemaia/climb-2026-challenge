"""Reusable ColonStreamSfSNet metric-depth inference.

Stateful model: a Mamba2 block carries recurrent state across frames, so
depth at frame k depends on every frame before it. Call `start_sequence()`
once before the first frame of a sequence, then `predict_depth_mm()` once per
frame, in order -- skipping frames or reusing the model on a new sequence
without resetting silently produces wrong depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model.sfsnet import SfSNet, build_model

# Set on the model instance by start_sequence(); checked by predict_depth_mm.
_SEQUENCE_FLAG = "_cs_sequence_started"

# ImageNet normalization stats, required to match the pretrained EfficientNetB0 backbone.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device,
    *,
    half_features: bool = True,
    temporal_layers: Sequence[str] = ("up4",),
    num_mamba_layers: int = 1,
    mamba_d_state: str | int = "d_model",
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    mamba_downsample: float = 0.1,
    max_depth: float = 100.0,
    gamma: float | None = 1.786087514758902,
    disparity: bool = False,
    resolution: Tuple[int, int] = (480, 640),
) -> Tuple[SfSNet, Dict]:
    """Build the model and load a checkpoint.

    Returns `(model, decode_cfg)`. `decode_cfg` carries the
    depth-decode keys (`max_depth`, `gamma`, `disparity`, `resolution`).
    """
    temporal_layers = list(temporal_layers)
    valid_levels = {"up1", "up2", "up3", "up4"}
    if not set(temporal_layers) <= valid_levels:
        raise ValueError(
            f"temporal_layers must be a subset of {sorted(valid_levels)}, got {temporal_layers}"
        )

    model_cfg = {
        "pretrained": False,
        "half_features": half_features,
        "temporal_layers": temporal_layers,
        "num_mamba_layers": num_mamba_layers,
        "mamba_d_state": mamba_d_state,
        "mamba_d_conv": mamba_d_conv,
        "mamba_expand": mamba_expand,
        "mamba_downsample": mamba_downsample,
    }
    model = build_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # A freshly loaded model has no sequence started yet.
    setattr(model, _SEQUENCE_FLAG, False)

    decode_cfg = {
        "max_depth": float(max_depth),
        "gamma": gamma,
        "disparity": bool(disparity),
        "resolution": (int(resolution[0]), int(resolution[1])),
    }
    return model, decode_cfg


def start_sequence(model: SfSNet) -> None:
    """Reset recurrent state. Call once before the first frame of a sequence."""
    model.start_new_sequence()
    setattr(model, _SEQUENCE_FLAG, True)


def decode_to_mm(
    pred: torch.Tensor,
    max_depth: float,
    gamma: float | None,
    disparity: bool,
) -> torch.Tensor:
    """Inverse of the training-time depth encoding.
    """
    if disparity:
        metric = torch.clamp(max_depth / (pred + 1e-8), 0, max_depth)
    else:
        metric = torch.clamp(pred, 0, max_depth)
    if gamma is not None:
        depth_01 = torch.clamp(metric / max_depth, 0.0, 1.0)
        metric = (depth_01 ** gamma) * max_depth
    return metric


def invalid_pixel_mask(frame_gpu: torch.Tensor, black_thresh: int = 10,
                        specular_thresh: "int | None" = None,
                        dilate_px: int = 0) -> torch.Tensor:
    """(H, W) bool mask of black-border pixels (max(R,G,B) <= black_thresh),
    optionally OR'd with specular-highlight pixels (min(R,G,B) >=
    specular_thresh) and dilated by `dilate_px`. `frame_gpu` is (H, W, 3)
    uint8 on the device; channel order doesn't matter.
    """
    mx, _ = frame_gpu.max(dim=-1)
    mask = mx <= black_thresh
    if specular_thresh is not None:
        mn, _ = frame_gpu.min(dim=-1)
        mask = mask | (mn >= specular_thresh)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        mask = F.max_pool2d(mask.float().unsqueeze(0).unsqueeze(0), k,
                             stride=1, padding=dilate_px).squeeze(0).squeeze(0) > 0.5
    return mask


@torch.no_grad()
def predict_depth_mm(
    model: SfSNet,
    frame: torch.Tensor,
    decode_cfg: Dict,
    invalid_mask: "torch.Tensor | None" = None,
    autocast: bool = False,
    mean_t: torch.Tensor | None = None,
    std_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one frame and advance the temporal state.

    `invalid_mask` ((H, W) bool, built by the caller, e.g. `invalid_pixel_mask`)
    zeroes those pixels in the output. If `None`, no masking is applied (e.g. warmup).
    """
    if not getattr(model, _SEQUENCE_FLAG, False):
        raise RuntimeError(
            "start_sequence(model) must be called before the first frame of a "
            "sequence. This model carries recurrent state across frames; running "
            "without a reset silently produces depth conditioned on the previous "
            "sequence."
        )

    orig_h, orig_w = frame.shape[:2]
    img_t = frame.permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Resized to trained model resolution, then normalized to ImageNet stats.
    if (orig_h, orig_w) != decode_cfg["resolution"]:
        target_h, target_w = decode_cfg["resolution"]
        img_t = F.interpolate(img_t, size=(target_h, target_w), mode="bilinear",
                               align_corners=False)
    img_t = (img_t - mean_t) / std_t

    if autocast:
        with torch.amp.autocast("cuda", enabled=True):
            pred = model(img_t)
    else:
        pred = model(img_t)

    depth = decode_to_mm(
        pred.float(),
        decode_cfg["max_depth"],
        decode_cfg["gamma"],
        decode_cfg["disparity"],
    )

    # Native output is half the model input resolution.
    depth_t = F.interpolate(depth, size=(orig_h, orig_w), mode="bilinear",
                             align_corners=False).squeeze(0).squeeze(0)
    if invalid_mask is not None:
        depth_t = depth_t.masked_fill(invalid_mask.to(device=depth_t.device, dtype=torch.bool), 0.0)
    return depth_t.float()


class ColonStreamSfSNetDepth:
    """Per-sequence streaming depth source: load once, `reset_state()` once
    per sequence, `estimate()` once per frame in order.
    """

    def __init__(self, ckpt_path, device='cuda:0', autocast=False, **model_kwargs):
        self.model, self.decode_cfg = load_model(ckpt_path, device, **model_kwargs)
        self.device = device
        self.autocast = bool(autocast)
        self._mean_t = torch.from_numpy(_IMAGENET_MEAN).view(3, 1, 1).to(device)
        self._std_t = torch.from_numpy(_IMAGENET_STD).view(3, 1, 1).to(device)

    def reset_state(self):
        start_sequence(self.model)

    def estimate(self, frame, invalid_mask=None):
        return predict_depth_mm(
            self.model,
            frame,
            self.decode_cfg,
            invalid_mask,
            autocast=self.autocast,
            mean_t=self._mean_t,
            std_t=self._std_t,
        )
