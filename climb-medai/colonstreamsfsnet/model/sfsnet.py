"""
Architecture: Pre-trained Encoder + lightweight Decoder with skip connections.

Based on:
Ruano, J., Gómez, M., Romero, E., & Manzanera, A. (2024). Leveraging a
realistic synthetic database to learn shape-from-shading for estimating the
colon depth in colonoscopy images. Computerized Medical Imaging and Graphics,
115, 102390.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import List, Dict, Union

from .temporal_mamba import TemporalMambaBlock


# Upsampling block

class UpProjectBlock(nn.Module):
    """
    Bilinear upsample ×2  →  concat(skip)  →  Conv-LReLU  →  Conv-LReLU
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels + skip_channels, out_channels,
                                kernel_size=3, padding=1, bias=True)
        self.conv_b = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        # Crop skip if there is a dimension mismatch (can occur at borders)
        if x.shape != skip.shape:
            skip = skip[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.conv_a(x))
        x = self.act(self.conv_b(x))
        return x


# Encoders

class EfficientNetB0Encoder(nn.Module):
    """EfficientNetB0 encoder. Skips: up1=240ch(H/16), up2=144ch(H/8),
    up3=96ch(H/4), up4=32ch(H/2). out_channels=1280."""

    SKIP_CHANNELS = [240, 144, 96, 32]   # up1..up4

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = tv_models.efficientnet_b0(weights=weights)
        self.features = base.features  # Sequential of MBConv blocks

        self._skips: Dict[str, torch.Tensor] = {}
        self._hooks = []

        # Attach hooks to grab intermediate feature maps for the decoder's
        # skip connections. Each one must match the decoder's resolution at
        # that stage, or it will be misaligned.
        self._register_hook(self.features[4][0].block[1], "up1")  # block4a act  → 240ch, H/16
        self._register_hook(self.features[3][0].block[1], "up2")  # block3a act  → 144ch, H/8
        self._register_hook(self.features[2][0].block[1], "up3")  # block2a act  →  96ch, H/4
        self._register_hook(self.features[0][2],          "up4")  # stem SiLU    →  32ch, H/2

        self.out_channels = 1280

    def _register_hook(self, layer: nn.Module, name: str):
        def hook(module, inp, out):
            self._skips[name] = out
        self._hooks.append(layer.register_forward_hook(hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, x: torch.Tensor):
        self._skips.clear()
        out = self.features(x)
        return out, dict(self._skips)

    def skip_channels(self) -> List[int]:
        return self.SKIP_CHANNELS


# Full SfSNet

class SfSNet(nn.Module):
    """Shape-from-Shading Network: EfficientNetB0 encoder + decoder with
    skip connections. Outputs a scalar depth map at half resolution."""

    def __init__(
        self,
        pretrained: bool = False,
        half_features: bool = True,
        temporal_layers: List[str] = [],
        num_mamba_layers: int = 1,
        mamba_d_state: Union[int, Dict[str, Union[int, str]]] = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_downsample: Union[float, Dict[str, float]] = 0.1,
    ):
        super().__init__()
        self.encoder = EfficientNetB0Encoder(pretrained=pretrained)
        enc_out = self.encoder.out_channels
        skip_ch = self.encoder.skip_channels()  # [up1, up2, up3, up4]

        # First decoder filter (equivalent to conv2 in Keras)
        dec0_out = enc_out // 2 if half_features else enc_out
        self.bottleneck = nn.Conv2d(enc_out, dec0_out, kernel_size=1, padding=0)

        # 4 upsampling blocks
        self.up1 = UpProjectBlock(dec0_out,       skip_ch[0], dec0_out // 2)
        self.up2 = UpProjectBlock(dec0_out // 2,  skip_ch[1], dec0_out // 4)
        self.up3 = UpProjectBlock(dec0_out // 4,  skip_ch[2], dec0_out // 8)
        self.up4 = UpProjectBlock(dec0_out // 8,  skip_ch[3], dec0_out // 16)

        # Final layer (equivalent to conv3 in Keras)
        self.final_conv = nn.Conv2d(dec0_out // 16, 1, kernel_size=3, padding=1)

        # Optional per-level temporal Mamba blocks
        level_channels = {
            "up1": dec0_out // 2,
            "up2": dec0_out // 4,
            "up3": dec0_out // 8,
            "up4": dec0_out // 16,
        }
        assert set(temporal_layers) <= set(level_channels), \
            f"temporal_layers must be a subset of {list(level_channels)}"
        if isinstance(mamba_downsample, dict):
            assert set(temporal_layers) <= set(mamba_downsample), \
                f"mamba_downsample dict must have an entry for every active level in {temporal_layers}, got keys {list(mamba_downsample)}"
        if isinstance(mamba_d_state, dict):
            assert set(temporal_layers) <= set(mamba_d_state), \
                f"mamba_d_state dict must have an entry for every active level in {temporal_layers}, got keys {list(mamba_d_state)}"

        def _resolve_d_state(name: str) -> int:
            value = mamba_d_state[name] if isinstance(mamba_d_state, dict) else mamba_d_state
            return level_channels[name] if value == "d_model" else int(value)

        self.temporal_blocks = nn.ModuleDict({
            name: TemporalMambaBlock(
                d_model=level_channels[name],
                num_layers=num_mamba_layers,
                d_state=_resolve_d_state(name),
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                downsample_factor=mamba_downsample[name] if isinstance(mamba_downsample, dict) else mamba_downsample,
            )
            for name in temporal_layers
        })

    def start_new_sequence(self):
        """Reset the state of every active temporal block before a new video."""
        for block in self.temporal_blocks.values():
            block.start_new_sequence()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input  : (B, 3, H, W)   — normalized RGB image [0,1]
        Output : (B, 1, H/2, W/2) — depth map
        """
        
        # Encoder
        enc_out, skips = self.encoder(x)

        # Bottleneck
        d = self.bottleneck(enc_out)

        # Decoder with skip connections
        d = self.up1(d, skips["up1"])
        if "up1" in self.temporal_blocks:
            d = self.temporal_blocks["up1"](d)
        d = self.up2(d, skips["up2"])
        if "up2" in self.temporal_blocks:
            d = self.temporal_blocks["up2"](d)
        d = self.up3(d, skips["up3"])
        if "up3" in self.temporal_blocks:
            d = self.temporal_blocks["up3"](d)
        d = self.up4(d, skips["up4"])
        if "up4" in self.temporal_blocks:
            d = self.temporal_blocks["up4"](d)

        # Final depth map (no activation — free regression)
        depth = self.final_conv(d)
        return depth

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Input  : (B, T, 3, H, W) — a full video clip, already on the target device
        Output : (B, T, 1, H/2, W/2) — depth map per frame

        Runs the model frame-by-frame so temporal blocks see prior frames'
        state. If no temporal blocks are active, the whole clip is processed
        at once instead (faster, same result).
        """
        B, T = x_seq.shape[:2]
        # Unconditional: every call starts a fresh clip. A no-op when there are
        # no temporal blocks (start_new_sequence iterates an empty ModuleDict),
        # so it costs nothing and keeps the invariant true on both paths.
        self.start_new_sequence()
        if not self.temporal_blocks:
            out = self.forward(x_seq.flatten(0, 1))
            return out.reshape(B, T, *out.shape[1:])
        preds = [self.forward(x_seq[:, t]) for t in range(T)]
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def forward_scene(self, x_seq: torch.Tensor, device, chunk_size: int = 8) -> torch.Tensor:
        """
        Input  : (1, T, 3, H, W) — one whole scene, may still be on CPU
        Output : (1, T, 1, H/2, W/2) — depth map per frame, on `device`

        Same as forward_sequence, but processes the scene in chunks of
        `chunk_size` frames to limit GPU memory use on long scenes. Temporal
        state is reset once for the whole scene, so chunking does not affect
        the result.
        """
        self.start_new_sequence()   # once per scene -- never inside the loop below
        T = x_seq.shape[1]
        out = []
        for start in range(0, T, chunk_size):
            chunk = x_seq[:, start:start + chunk_size].to(device, non_blocking=True)
            if self.temporal_blocks:
                # Call forward() directly, not forward_sequence(), which would
                # reset state and restart the scene at each chunk.
                preds = [self.forward(chunk[:, t]) for t in range(chunk.shape[1])]
                out.append(torch.stack(preds, dim=1))
            else:
                B, T_chunk = chunk.shape[:2]
                o = self.forward(chunk.flatten(0, 1))
                out.append(o.reshape(B, T_chunk, *o.shape[1:]))
            del chunk
        return torch.cat(out, dim=1)


# Factory

def build_model(cfg: dict) -> SfSNet:
    """
    Builds the model from the 'model' block of config.yaml.
    """
    return SfSNet(
        pretrained=cfg.get("pretrained", False),
        half_features=cfg.get("half_features", True),
        temporal_layers=cfg.get("temporal_layers", []),
        num_mamba_layers=cfg.get("num_mamba_layers", 1),
        mamba_d_state=cfg.get("mamba_d_state", 64),
        mamba_d_conv=cfg.get("mamba_d_conv", 4),
        mamba_expand=cfg.get("mamba_expand", 2),
        mamba_downsample=cfg.get("mamba_downsample", 0.1),
    )
