"""
Kiến trúc CNN tối ưu cho bài toán Nhận dạng Cảm xúc Khuôn mặt KDEF.
Sử dụng duy nhất kiến trúc chiến thắng: KDEF_Stem2Ch_ECA_Strip (1.239.597 tham số).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class ECALayer(nn.Module):
    """ECA-Net: Efficient Channel Attention cho CNN (Wang et al., CVPR 2020)."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class KDEF_Stem2Ch_ECA_Strip(nn.Module):
    """
    Kiến trúc CNN Nâng cấp Sâu cho KDEF (1.239.597 tham số):
    - Multi-Scale Parallel Stem: Conv 3x3 (1->16) + Conv 5x5 (1->16) -> 32 kênh
    - 4 Stage phân cấp với hàm kích hoạt SiLU mượt mà (0.00% neuron chết)
    - Chú ý kênh siêu nhẹ ECA-Net (kernel_size=3) tại Stage 3 và Stage 4
    - Upper/Lower Facial Strip Pooling AdaptiveAvgPool2d((2, 1)) tách Action Units mày & miệng
    - Classifier Head 128-dim chuẩn hóa BatchNorm1d + Dropout(0.4, 0.3)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 7):
        super().__init__()
        self.model_name = "kdef_stem2ch_eca_strip"
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem song song 3x3 + 5x5
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

        # Stage 3: 64 -> 128 -> 128 + ECA-Net
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

        # Stage 4: 128 -> 256 -> 256 + ECA-Net
        self.stage4_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.stage4_bn1 = nn.BatchNorm2d(256)
        self.stage4_act1 = nn.SiLU(inplace=True)
        self.stage4_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.stage4_bn2 = nn.BatchNorm2d(256)
        self.stage4_act2 = nn.SiLU(inplace=True)
        self.stage4_eca = ECALayer(kernel_size=3)

        # Upper/Lower Strip Pooling & Head
        self.strip_pool = nn.AdaptiveAvgPool2d((2, 1))
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
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
        """Trích xuất feature map Stage 4 trước strip pooling (dùng cho Grad-CAM & phân tích tầng)."""
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
        """Đo thống kê activation qua 4 Stage (dead neuron & độ lệch chuẩn)."""
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


def build_model(
    model_name: str = "kdef_stem2ch_eca_strip",
    num_classes: int = 7,
    in_channels: int = 1,
) -> nn.Module:
    """Khởi tạo kiến trúc chuẩn duy nhất KDEF_Stem2Ch_ECA_Strip."""
    name = model_name.lower().replace("-", "_")
    if name in ("kdef_stem2ch_eca_strip", "stem2ch", "kdef_stem2ch", "final"):
        return KDEF_Stem2Ch_ECA_Strip(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(
        f"Kiến trúc không được hỗ trợ: {model_name}. Chỉ sử dụng 'kdef_stem2ch_eca_strip'."
    )


def load_kdef_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    num_classes: int = 7,
) -> Tuple[nn.Module, Dict]:
    """Nạp checkpoint KDEF_Stem2Ch_ECA_Strip và trả về (model_eval, metadata)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state" in ckpt):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state")
    else:
        state_dict = ckpt
        ckpt = {}

    if "stem_3x3.weight" not in state_dict:
        raise ValueError(
            f"Checkpoint {checkpoint_path} không khớp kiến trúc KDEF_Stem2Ch_ECA_Strip."
        )

    model = KDEF_Stem2Ch_ECA_Strip(in_channels=1, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    norm = ckpt.get("normalization", {})
    cfg = ckpt.get("config", {})
    meta = {
        "model_name": "kdef_stem2ch_eca_strip",
        "mean": float(norm.get("mean", 0.4056757390499115)),
        "std": float(norm.get("std", 0.24074745178222656)),
        "use_crop": bool(cfg.get("crop", True)),
        "use_clahe": bool(cfg.get("clahe", True)),
        "img_size": 128,
        "epoch": ckpt.get("epoch"),
        "val_macro_f1": ckpt.get("val_macro_f1", ckpt.get("val_f1")),
    }
    return model, meta


def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số tham số có thể huấn luyện."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
