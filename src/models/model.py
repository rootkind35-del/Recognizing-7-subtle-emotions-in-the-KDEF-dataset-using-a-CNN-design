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
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
# 5. Model Factory & Checkpoint Loader
# ---------------------------------------------------------------------------
def build_model(
    model_name: str = "kdef_res_se_net",
    num_classes: int = 7,
    in_channels: int = 1,
    **kwargs
) -> nn.Module:
    """
    Constructs and returns the specified emotion recognition model.
    Supported model_name values:
      - 'kdef_res_se_net' (default, ~1.34M params)
      - 'kdefnet' (4-block CNN, ~1.24M params)
    """
    name = model_name.lower().replace("-", "_")
    if name in ["kdef_res_se_net", "res_se_net", "optimized", "kdefressen"]:
        return KDEFResSENet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif name in ["kdefnet", "baseline_kdefnet", "notebook"]:
        return KDEFNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Available: 'kdef_res_se_net', 'kdefnet'")


def load_kdef_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    num_classes: int = 7
) -> Tuple[nn.Module, Dict]:
    """
    Loads either an optimized core checkpoint (checkpoints/optimized_best_model.pt)
    or a notebook checkpoint (checkpoints/final_model.pt, baseline.pt, etc.)
    and returns (model_in_eval_mode, metadata_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state" in ckpt):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state")
    else:
        state_dict = ckpt
        ckpt = {}

    is_notebook_ckpt = "model_state" in ckpt or "features.0.0.bias" in state_dict
    model_name = ckpt.get("model_name", "kdefnet" if is_notebook_ckpt else "kdef_res_se_net")

    if model_name.lower().replace("-", "_") in ["kdefnet", "baseline_kdefnet", "notebook"]:
        conv_bias = "features.0.0.bias" in state_dict
        head_bn = "head.2.running_mean" in state_dict
        model = KDEFNet(in_channels=1, num_classes=num_classes, conv_bias=conv_bias, head_bn=head_bn)
    else:
        model = build_model(model_name=model_name, num_classes=num_classes, in_channels=1)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    norm = ckpt.get("normalization", {})
    cfg = ckpt.get("config", {})
    meta = {
        "model_name": model_name,
        "is_notebook_checkpoint": is_notebook_ckpt,
        "mean": float(norm.get("mean", 0.3370 if is_notebook_ckpt else 0.360891)),
        "std": float(norm.get("std", 0.1871 if is_notebook_ckpt else 0.210004)),
        "use_crop": bool(cfg.get("crop")) if not is_notebook_ckpt else False,
        "use_clahe": bool(cfg.get("clahe")) if not is_notebook_ckpt else False,
        "img_size": 128,
        "epoch": ckpt.get("epoch"),
        "val_macro_f1": ckpt.get("val_macro_f1", ckpt.get("val_f1")),
    }
    return model, meta
