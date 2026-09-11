"""Per-level temporal Mamba block for SfSNet. `TemporalMambaBlock` packages
a downsample -> scan -> upsample -> residual pattern as a standalone module,
one instance per decoder level.

Uses the Mamba2 layer from:
Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models and
efficient algorithms through structured state space duality. arXiv preprint
arXiv:2405.21060.

`InferenceParams`/`MambaBlock` are adapted from:
Li, H., Lu, D., Wang, J., Webster III, R. J., & Oguz, I. (2026).
EndoStreamDepth: Temporally consistent monocular depth estimation for
endoscopic video streams. Proceedings of Machine Learning Research, 315, 1697.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InferenceParams:
    """Persistent per-sequence state threaded through Mamba2 calls."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    seq_idx_dict: dict = field(default_factory=dict)

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0


class MambaBlock(nn.Module):
    """Pre-norm Mamba2 + MLP block."""

    def __init__(self, d_model, layer_idx, expand, d_state=64, d_conv=4, headdim=8):
        super().__init__()
        from mamba_ssm import Mamba2

        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
            headdim=headdim,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, inference_params=None):
        x = x + self.mamba(self.norm1(x), inference_params=inference_params)
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalMambaBlock(nn.Module):
    """One temporal Mamba block for a single SfSNet decoder level: pools
    spatial features to a small token grid, runs them through Mamba2 with
    persistent state across frames, upsamples back, and adds as a residual.

    Call `start_new_sequence()` before the first frame of a new video;
    `forward()` does not reset state on its own.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 8,
        downsample_factor: float = 0.1,
        max_batch_size: int = 32,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, layer_idx=i, expand=expand, d_state=d_state,
                       d_conv=d_conv, headdim=headdim)
            for i in range(num_layers)
        ])
        self.final_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.final_layer[1].weight)
        nn.init.zeros_(self.final_layer[1].bias)

        self.downsample_factor = downsample_factor
        self.max_seqlen = 60000
        self.max_batch_size = max_batch_size
        self.inference_params = InferenceParams(
            max_seqlen=self.max_seqlen, max_batch_size=self.max_batch_size
        )

    def start_new_sequence(self):
        """Reset hidden state before the first frame of a new video."""
        self.inference_params = InferenceParams(
            max_seqlen=self.max_seqlen, max_batch_size=self.max_batch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, h, w) -> same shape, temporal correction added as a
        residual. Call once per frame, in order."""
        original = x
        B, C, h, w = x.shape

        h_ds = max(int(round(h * self.downsample_factor)), 1)
        w_ds = max(int(round(w * self.downsample_factor)), 1)
        pooled = F.adaptive_avg_pool2d(x, (h_ds, w_ds))

        tokens = pooled.flatten(2).transpose(1, 2)  # (B, h_ds*w_ds, C)
        for block in self.blocks:
            tokens = block(tokens, inference_params=self.inference_params)
        self.inference_params.seqlen_offset += tokens.shape[1]

        spatial = tokens.transpose(1, 2).reshape(B, C, h_ds, w_ds)
        spatial = F.interpolate(spatial, size=(h, w), mode="bilinear", align_corners=True)

        tokens_full = spatial.flatten(2).transpose(1, 2)  # (B, h*w, C)
        tokens_full = self.final_layer(tokens_full)
        correction = tokens_full.transpose(1, 2).reshape(B, C, h, w)

        return original + correction
