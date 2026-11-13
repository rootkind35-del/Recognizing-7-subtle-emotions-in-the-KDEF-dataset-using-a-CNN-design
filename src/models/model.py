"""
Neural Network Architectures for KDEF Facial Emotion Recognition.
Provides:
- KDEFResSENet: Optimized core architecture with Residual Skip Connections,
  Squeeze-and-Excitation (SE) Channel Attention, smooth SiLU activations,
  and Dual Feature Pooling (Avg + Max). Total params: ~1.34M.
- KDEFNet: 1.24M 4-block CNN architecture (compatible with both optimized core
  and notebooks/CNN_KDEF_FER.ipynb checkpoints).
- build_model & load_kdef_checkpoint: Universal model factory and checkpoint loader.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Squeeze-and-Excitation (SE) Channel Attention Block
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention (Hu et al., 2018).
    Adaptively recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ---------------------------------------------------------------------------
# 2. Residual Squeeze-and-Excitation Convolution Block
# ---------------------------------------------------------------------------
class ResSEBlock(nn.Module):
    """
    Two-layer 3x3 Conv block with BatchNorm, smooth non-linear activation (SiLU),
    residual skip connection, SE channel attention, and spatial pooling.
    """
    def __init__(
        self,
        in_c: int,
        out_c: int,
        act_type: str = "silu",
        use_se: bool = True,
        drop2d: float = 0.0,
        pool: bool = True
    ):
        super().__init__()
        self.use_se = use_se
        self.pool = pool

        def make_act():
            if act_type == "silu":
                return nn.SiLU(inplace=True)
            elif act_type == "leaky_relu":
                return nn.LeakyReLU(0.1, inplace=True)
            elif act_type == "gelu":
                return nn.GELU()
            return nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = make_act()

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c, reduction=8) if use_se else nn.Identity()

        # Shortcut projection if channel dimension changes
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = make_act()
        self.drop = nn.Dropout2d(drop2d) if drop2d > 0.0 else nn.Identity()
        self.maxpool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward_pre_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Returns activation right before pooling for diagnostic probe."""
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        out = self.act2(out)
        out = self.drop(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward_pre_pool(x)
        return self.maxpool(out)


# ---------------------------------------------------------------------------
# 3. KDEFResSENet (Winning Optimized Core Architecture)
# ---------------------------------------------------------------------------
class KDEFResSENet(nn.Module):
    """
    KDEFResSENet: Optimized Facial Emotion Recognition Architecture.
    Features:
    - 4 stages of Residual Squeeze-and-Excitation blocks: [32, 64, 128, 256]
    - Smooth SiLU activations completely eliminating Stage 4 dead neurons
    - Dual feature pooling (AdaptiveAvgPool2d + AdaptiveMaxPool2d) preserving both
      global expression tone and sharp Action Unit (AU) muscle contractions
    - Regularized classifier head with BatchNorm1d, SiLU, and Dropout (0.4, 0.3)
    - Total parameters: ~1.34M.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        act_type: str = "silu",
        use_se: bool = True,
        drop2d: float = 0.05,
        head_p1: float = 0.4,
        head_p2: float = 0.3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        stages = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            # Apply mild 2D spatial dropout on deep stages (stages 2 and 3)
            d2 = drop2d if i >= 2 else 0.0
            stages.append(ResSEBlock(
                in_c=c_in,
                out_c=c_out,
                act_type=act_type,
                use_se=use_se,
                drop2d=d2,
                pool=True
            ))
            c_in = c_out
        self.features = nn.Sequential(*stages)

        # Dual pooling: combines global spatial average and peak muscle activation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        feat_dim = channels[-1] * 2  # 256 * 2 = 512

        self.head = nn.Sequential(
            nn.Dropout(head_p1),
            nn.Linear(feat_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True) if act_type == "silu" else nn.ReLU(inplace=True),
            nn.Dropout(head_p2),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def extract_stage4_pre_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature map of Stage 4 (Block 4) right before MaxPool2d.
        Directly corresponds to `features.3.5` activation probe in baseline KDEFNet.
        """
        s0 = self.features[0](x)
        s1 = self.features[1](s0)
        s2 = self.features[2](s1)
        blk3 = self.features[3]
        return blk3.forward_pre_pool(s2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        avg_f = self.avg_pool(f).flatten(1)
        max_f = self.max_pool(f).flatten(1)
        return torch.cat([avg_f, max_f], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.forward_features(x)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# 4. KDEFNet (1.24M 4-Block CNN Architecture)
# ---------------------------------------------------------------------------
class KDEFNet(nn.Module):
    """
    4-block CNN architecture for KDEF (1.24M params).
    Supports both the optimized core variant (`conv_bias=False, head_bn=True`)
    and the notebook variant (`conv_bias=True, head_bn=False`).
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        conv_bias: bool = False,
        head_bn: bool = True
    ):
        super().__init__()
        def make_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=conv_bias),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=conv_bias),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

        self.features = nn.Sequential(
            make_block(in_channels, 32),   # features.0
            make_block(32, 64),            # features.1
            make_block(64, 128),           # features.2
            make_block(128, 256)           # features.3
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        if head_bn:
            self.head = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(256, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(256, 256, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

    def extract_stage4_pre_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features.3.5 activation for diagnostic probe."""
        s0 = self.features[0](x)
        s1 = self.features[1](s0)
        s2 = self.features[2](s1)
        blk = self.features[3]
        out = blk[2](blk[1](blk[0](s2)))  # Conv -> BN -> ReLU
        out = blk[5](blk[4](blk[3](out))) # Conv -> BN -> ReLU (features.3.5)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        p = self.gap(f).flatten(1)
        return self.head(p)


# ---------------------------------------------------------------------------
# 5. ECA-Net & KDEF_Stem2Ch_ECA_Strip (Phase 2 Winning Architecture)
# ---------------------------------------------------------------------------
class ECALayer(nn.Module):
    """
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (Wang et al., CVPR 2020).
    Captures local cross-channel interaction via 1D convolution of size k without dimensionality reduction.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class KDEF_Stem2Ch_ECA_Strip(nn.Module):
    """
    Winning Phase 2 Deep Upgraded CNN Architecture (1,239,597 parameters):
    - Parallel 3x3 + 5x5 Multi-Scale Stem (16+16=32ch)
    - 4-stage hierarchy with SiLU activations (0.0000 exact dead neurons, 2.08x std ratio)
    - Lightweight ECA-Net attention (k=3) in stages 3 and 4
    - Upper/Lower Facial Strip Pooling AdaptiveAvgPool2d((2, 1)) separating brow vs mouth AUs
    - Compact 128-dim projection head with zero parameter bloat
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 7):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Parallel Stem: 3x3 and 5x5 filters
        self.stem_3x3 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.stem_5x5 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2, bias=False)
        self.stem_bn1 = nn.BatchNorm2d(32)
        self.stem_act1 = nn.SiLU(inplace=True)
        self.stem_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.stem_bn2 = nn.BatchNorm2d(32)
        self.stem_act2 = nn.SiLU(inplace=True)
        self.stem_pool = nn.MaxPool2d(2, 2)

        # Stage 2: 32 -> 64 -> 64
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Stage 3: 64 -> 128 -> 128 with ECA-Net
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            ECALayer(kernel_size=3),
            nn.MaxPool2d(2, 2),
        )

        # Stage 4: 128 -> 256 -> 256 with ECA-Net
        self.stage4_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.stage4_bn1 = nn.BatchNorm2d(256)
        self.stage4_act1 = nn.SiLU(inplace=True)
        self.stage4_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.stage4_bn2 = nn.BatchNorm2d(256)
        self.stage4_act2 = nn.SiLU(inplace=True)
        self.stage4_eca = ECALayer(kernel_size=3)

        # Facial Upper/Lower Strip Pooling (brow vs mouth AUs)
        self.strip_pool = nn.AdaptiveAvgPool2d((2, 1))

        # Compact Classification Head: 256 * 2 * 1 = 512 -> 128 -> 7
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_stage4_pre_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts Stage 4 feature map right before strip pooling (for Grad-CAM and diagnostics)."""
        c3 = self.stem_3x3(x)
        c5 = self.stem_5x5(x)
        s1 = torch.cat([c3, c5], dim=1)
        s1 = self.stem_act1(self.stem_bn1(s1))
        s1 = self.stem_pool(self.stem_act2(self.stem_bn2(self.stem_conv2(s1))))

        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        out = self.stage4_act1(self.stage4_bn1(self.stage4_conv1(s3)))
        out = self.stage4_act2(self.stage4_bn2(self.stage4_conv2(out)))
        out = self.stage4_eca(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.extract_stage4_pre_pool(x)
        p = self.strip_pool(f).flatten(1)
        return self.head(p)

    def probe_activations(self, x: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Probes activations across Stages 1-4 for dead neuron and variance stats."""
        stats = {}
        with torch.no_grad():
            c3 = self.stem_3x3(x)
            c5 = self.stem_5x5(x)
            s1 = torch.cat([c3, c5], dim=1)
            s1 = self.stem_act1(self.stem_bn1(s1))
            s1_pre = self.stem_act2(self.stem_bn2(self.stem_conv2(s1)))
            flat1 = s1_pre.detach().cpu().flatten().numpy()
            stats["stage1"] = {
                "mean": float(np.mean(flat1)),
                "std": float(np.std(flat1)),
                "exact_dead_frac": float(np.mean(np.abs(flat1) < 1e-6)),
                "neg_frac": float(np.mean(flat1 <= 0.0)),
            }
            s1 = self.stem_pool(s1_pre)

            s2_c1 = self.stage2[2](self.stage2[1](self.stage2[0](s1)))
            s2_pre = self.stage2[5](self.stage2[4](self.stage2[3](s2_c1)))
            flat2 = s2_pre.detach().cpu().flatten().numpy()
            stats["stage2"] = {
                "mean": float(np.mean(flat2)),
                "std": float(np.std(flat2)),
                "exact_dead_frac": float(np.mean(np.abs(flat2) < 1e-6)),
                "neg_frac": float(np.mean(flat2 <= 0.0)),
            }
            s2 = self.stage2[6](s2_pre)

            s3_c1 = self.stage3[2](self.stage3[1](self.stage3[0](s2)))
            s3_pre = self.stage3[5](self.stage3[4](self.stage3[3](s3_c1)))
            s3_pre = self.stage3[6](s3_pre)
            flat3 = s3_pre.detach().cpu().flatten().numpy()
            stats["stage3"] = {
                "mean": float(np.mean(flat3)),
                "std": float(np.std(flat3)),
                "exact_dead_frac": float(np.mean(np.abs(flat3) < 1e-6)),
                "neg_frac": float(np.mean(flat3 <= 0.0)),
            }
            s3 = self.stage3[7](s3_pre)

            s4_pre = self.stage4_act1(self.stage4_bn1(self.stage4_conv1(s3)))
            s4_pre = self.stage4_act2(self.stage4_bn2(self.stage4_conv2(s4_pre)))
            s4_pre = self.stage4_eca(s4_pre)
            flat4 = s4_pre.detach().cpu().flatten().numpy()
            stats["stage4"] = {
                "mean": float(np.mean(flat4)),
                "std": float(np.std(flat4)),
                "exact_dead_frac": float(np.mean(np.abs(flat4) < 1e-6)),
                "neg_frac": float(np.mean(flat4 <= 0.0)),
            }
        return stats


# ---------------------------------------------------------------------------
# 6. Model Factory & Checkpoint Loader
# ---------------------------------------------------------------------------
def build_model(
    model_name: str = "kdef_stem2ch_eca_strip",
    num_classes: int = 7,
    in_channels: int = 1,
    **kwargs
) -> nn.Module:
    """
    Constructs and returns the specified emotion recognition model.
    Supported model_name values:
      - 'kdef_stem2ch_eca_strip' (Phase 2 winner, 1,239,597 params)
      - 'kdef_res_se_net' (~1.34M params)
      - 'kdefnet' (4-block CNN, ~1.24M params)
    """
    name = model_name.lower().replace("-", "_")
    if name in ["kdef_stem2ch_eca_strip", "stem2ch", "phase2_winner", "kdef_stem2ch"]:
        return KDEF_Stem2Ch_ECA_Strip(in_channels=in_channels, num_classes=num_classes)
    elif name in ["kdef_res_se_net", "res_se_net", "optimized", "kdefressen"]:
        return KDEFResSENet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif name in ["kdefnet", "baseline_kdefnet", "notebook"]:
        return KDEFNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. Available: "
            f"'kdef_stem2ch_eca_strip', 'kdef_res_se_net', 'kdefnet'"
        )


def load_kdef_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    num_classes: int = 7
) -> Tuple[nn.Module, Dict]:
    """
    Loads either a Phase 2 winning checkpoint (checkpoints/final_model.pt / optimized_best_model.pt),
    an optimized core checkpoint, or a legacy notebook checkpoint.
    Returns (model_in_eval_mode, metadata_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state" in ckpt):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state")
    else:
        state_dict = ckpt
        ckpt = {}

    model_name = ckpt.get("model_name", "")
    is_stem2ch = "stem_3x3.weight" in state_dict or model_name.lower().replace("-", "_") in [
        "kdef_stem2ch_eca_strip", "stem2ch", "phase2_winner", "kdef_stem2ch"
    ]
    is_notebook_ckpt = "model_state" in ckpt or "features.0.0.bias" in state_dict

    if is_stem2ch:
        model = KDEF_Stem2Ch_ECA_Strip(in_channels=1, num_classes=num_classes)
        model_name = "kdef_stem2ch_eca_strip"
    elif model_name.lower().replace("-", "_") in ["kdefnet", "baseline_kdefnet", "notebook"] or is_notebook_ckpt:
        conv_bias = "features.0.0.bias" in state_dict
        head_bn = "head.2.running_mean" in state_dict
        model = KDEFNet(in_channels=1, num_classes=num_classes, conv_bias=conv_bias, head_bn=head_bn)
        model_name = "kdefnet"
    else:
        model = build_model(model_name=model_name or "kdef_res_se_net", num_classes=num_classes, in_channels=1)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    norm = ckpt.get("normalization", {})
    cfg = ckpt.get("config", {})
    meta = {
        "model_name": model_name,
        "is_notebook_checkpoint": is_notebook_ckpt,
        "mean": float(norm.get("mean", 0.407835 if is_stem2ch else (0.3370 if is_notebook_ckpt else 0.360891))),
        "std": float(norm.get("std", 0.242375 if is_stem2ch else (0.1871 if is_notebook_ckpt else 0.210004))),
        "use_crop": bool(cfg.get("crop", True)),
        "use_clahe": bool(cfg.get("clahe", True)),
        "img_size": 128,
        "epoch": ckpt.get("epoch"),
        "val_macro_f1": ckpt.get("val_macro_f1", ckpt.get("val_f1")),
    }
    return model, meta
