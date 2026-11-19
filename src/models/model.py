"""
Kiến trúc CNN tối ưu cho bài toán Nhận dạng Cảm xúc Khuôn mặt KDEF.
Sử dụng duy nhất kiến trúc chiến thắng: KDEF_Stem2Ch_MS4_Symmetric (`stem2ch_ms4_fuse`, 1.273.005 tham số).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SymmetricMultiScaleAUBlock(nn.Module):
    """
    Khối Stage 4 đa tỉ lệ đối xứng trái/phải nghiêm ngặt (100% L/R flip-equivariant):
    Song song 3x3 (nếp nhăn cục bộ) + (1,3) (mày/môi ngang) + (3,1) (rãnh mũi-má/hàm dọc) + 3x3_d2 (AU toàn cục),
    trộn 320->256 qua Conv 1x1 bảo toàn đầy đủ dung lượng biểu diễn + ECA-Net (k=3).
    Dùng SiLU(inplace=False) để tương thích an toàn tuyệt đối với Grad-CAM full backward hook.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        c33, c13, c31, cd2 = 96, 64, 64, 96
        mid_ch = c33 + c13 + c31 + cd2
        self.b33 = nn.Conv2d(channels, c33, kernel_size=3, padding=1, bias=False)
        self.b13 = nn.Conv2d(channels, c13, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.b31 = nn.Conv2d(channels, c31, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bd2 = nn.Conv2d(channels, cd2, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(mid_ch)
        self.act = nn.SiLU(inplace=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=False),
        )
        self.eca = ECALayer(kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b33(x), self.b13(x), self.b31(x), self.bd2(x)], dim=1)
        out = self.fuse(self.act(self.bn(out)))
        return self.eca(out)


class FacialProjectionHead(nn.Module):
    """Head 512 -> 128-dim L2-normalized embedding (cho SupCon) + classifier tuyến tính 7 lớp."""

    def __init__(
        self,
        in_features: int = 512,
        embed_dim: int = 128,
        num_classes: int = 7,
        drop_in: float = 0.45,
        drop_mid: float = 0.30,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(drop_in),
            nn.Linear(in_features, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.drop_mid = nn.Dropout(drop_mid)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self, pooled: torch.Tensor, return_embed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feat = self.proj(pooled)
        embed = F.normalize(feat, p=2, dim=1)
        logits = self.classifier(self.drop_mid(feat))
        if return_embed:
            return logits, embed
        return logits


def _compute_act_stats(tensor: torch.Tensor) -> Dict[str, float]:
    flat = tensor.detach().cpu().flatten().numpy()
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "exact_dead_frac": float(np.mean(np.abs(flat) < 1e-6)),
        "neg_frac": float(np.mean(flat <= 0.0)),
    }


class KDEF_Stem2Ch_MS4_Symmetric(nn.Module):
    """
    Kiến trúc CNN tối ưu KDEF_Stem2Ch_MS4_Symmetric (`stem2ch_ms4_fuse`, 1.273.005 tham số @ 1ch):
    - Multi-Scale Parallel Stem: Conv 3x3 (1->16) + Conv 5x5 (1->16) -> 32 kênh -> Conv 3x3 -> MaxPool2d
    - Stage 2-3 tuần tự chuẩn hóa (lọc nhiễu danh tính người mẫu) + ECA-Net tại Stage 3
    - Stage 4: Conv 128->256 + SymmetricMultiScaleAUBlock(256) đa tỉ lệ (3x3 + (1,3) + (3,1) + d=2 -> 1x1 320->256 + ECA)
    - Upper/Lower Facial Strip Pooling AdaptiveAvgPool2d((2, 1)) tách vùng mày/mắt và mũi/miệng
    - Head 512->128->7 hỗ trợ `forward(x, return_embed=True)` cho SupCon và `get_gradcam_target_layers()`
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 7, variant: str = "v1"):
        super().__init__()
        self.model_name = "stem2ch_ms4_fuse"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.variant = str(variant)

        # Stem song song 3x3 + 5x5 -> 32 -> 3x3 -> MaxPool
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
            nn.Identity(),
            nn.MaxPool2d(2, 2),
        )

        # Stage 3: 64 -> 128 -> 128 + ECA-Net
        self.s3_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.s3_bn1 = nn.BatchNorm2d(128)
        self.s3_act1 = nn.SiLU(inplace=True)
        self.s3_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.s3_bn2 = nn.BatchNorm2d(128)
        self.s3_eca = ECALayer(kernel_size=3)
        self.s3_act2 = nn.SiLU(inplace=True)
        self.s3_pool = nn.MaxPool2d(2, 2)

        # Stage 4: 128 -> 256 -> SymmetricMultiScaleAUBlock(256)
        self.s4_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.s4_bn1 = nn.BatchNorm2d(256)
        self.s4_act1 = nn.SiLU(inplace=True)
        self.s4_ms = SymmetricMultiScaleAUBlock(256)

        if self.variant == "v2_res":
            self.s2_skip = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
            )
            self.s3_skip = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
            )
            head_in_dim = 1024
            drop_in, drop_mid = 0.30, 0.15
        else:
            head_in_dim = 512
            drop_in, drop_mid = 0.45, 0.30

        # Upper/Lower Facial Strip Pooling & Head -> 128 -> 7
        self.head = FacialProjectionHead(
            in_features=head_in_dim,
            embed_dim=128,
            num_classes=num_classes,
            drop_in=drop_in,
            drop_mid=drop_mid,
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
        """Trích xuất feature map Stage 4 (256x20x20 với 160x160) trước strip pooling."""
        s1 = torch.cat([self.stem_3x3(x), self.stem_5x5(x)], dim=1)
        s1 = self.stem_act1(self.stem_bn1(s1))
        s1 = self.stem_pool(self.stem_act2(self.stem_bn2(self.stem_conv2(s1))))
        if self.variant == "v2_res":
            s2_h = self.stage2[2](self.stage2[1](self.stage2[0](s1)))
            s2_h = self.stage2[4](self.stage2[3](s2_h))
            s2 = self.stage2[7](self.stage2[5](s2_h + self.s2_skip(s1)))

            h3 = self.s3_act1(self.s3_bn1(self.s3_conv1(s2)))
            h3 = self.s3_eca(self.s3_bn2(self.s3_conv2(h3)))
            s3 = self.s3_pool(self.s3_act2(h3 + self.s3_skip(s2)))

            h4 = self.s4_act1(self.s4_bn1(self.s4_conv1(s3)))
            return self.s4_ms(h4) + h4

        s2 = self.stage2(s1)

        h3 = self.s3_act1(self.s3_bn1(self.s3_conv1(s2)))
        h3 = self.s3_act2(self.s3_eca(self.s3_bn2(self.s3_conv2(h3))))
        s3 = self.s3_pool(h3)

        h4 = self.s4_act1(self.s4_bn1(self.s4_conv1(s3)))
        return self.s4_ms(h4)

    def get_gradcam_target_layers(self) -> List[nn.Module]:
        """Trả về tầng đích Grad-CAM an toàn với full backward hook."""
        return [self.s4_ms.eca]

    def forward(
        self, x: torch.Tensor, return_embed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        f = self.extract_stage4_pre_pool(x)
        if self.variant == "v2_res":
            p_avg = F.adaptive_avg_pool2d(f, (2, 1))
            p_max = F.adaptive_max_pool2d(f, (2, 1))
            p = torch.cat([p_avg, p_max], dim=1).flatten(1)
        else:
            p = F.adaptive_avg_pool2d(f, (2, 1)).flatten(1)
        return self.head(p, return_embed=return_embed)

    def probe_activations(self, x: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Đo thống kê activation qua 4 Stage (dead neuron & độ lệch chuẩn)."""
        stats = {}
        with torch.no_grad():
            s1 = torch.cat([self.stem_3x3(x), self.stem_5x5(x)], dim=1)
            s1 = self.stem_act1(self.stem_bn1(s1))
            s1_pre = self.stem_act2(self.stem_bn2(self.stem_conv2(s1)))
            stats["stage1"] = _compute_act_stats(s1_pre)
            s1_p = self.stem_pool(s1_pre)

            s2_c1 = self.stage2[2](self.stage2[1](self.stage2[0](s1_p)))
            s2_pre = self.stage2[5](self.stage2[4](self.stage2[3](s2_c1)))
            stats["stage2"] = _compute_act_stats(s2_pre)
            s2 = self.stage2[7](self.stage2[6](s2_pre))

            h3 = self.s3_act1(self.s3_bn1(self.s3_conv1(s2)))
            h3 = self.s3_act2(self.s3_eca(self.s3_bn2(self.s3_conv2(h3))))
            stats["stage3"] = _compute_act_stats(h3)

            s4 = self.extract_stage4_pre_pool(x)
            stats["stage4"] = _compute_act_stats(s4)
        return stats


def build_model(
    model_name: str = "stem2ch_ms4_fuse",
    num_classes: int = 7,
    in_channels: int = 1,
) -> nn.Module:
    """Khởi tạo kiến trúc chuẩn duy nhất KDEF_Stem2Ch_MS4_Symmetric (`stem2ch_ms4_fuse`)."""
    name = model_name.lower().replace("-", "_")
    if name in (
        "stem2ch_ms4_fuse",
        "kdef_stem2ch_ms4_symmetric",
        "kdef_stem2ch_ms4_fuse",
        "final",
        "default",
    ):
        return KDEF_Stem2Ch_MS4_Symmetric(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(
        f"Kiến trúc không được hỗ trợ: {model_name}. Chỉ sử dụng 'stem2ch_ms4_fuse' (KDEF_Stem2Ch_MS4_Symmetric)."
    )


def load_kdef_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    num_classes: int = 7,
) -> Tuple[nn.Module, Dict]:
    """Nạp checkpoint KDEF_Stem2Ch_MS4_Symmetric và trả về `(model_eval, metadata)`."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state" in ckpt):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state")
    else:
        state_dict = ckpt
        ckpt = {}

    if "s4_ms.fuse.0.weight" not in state_dict:
        raise ValueError(
            f"Checkpoint {checkpoint_path} không khớp kiến trúc KDEF_Stem2Ch_MS4_Symmetric (thiếu 's4_ms.fuse.0.weight')."
        )

    norm = ckpt.get("normalization", {})
    cfg = ckpt.get("config", {})
    in_channels = int(cfg.get("in_channels", state_dict["stem_3x3.weight"].shape[1]))
    crop_mode = str(cfg.get("crop_mode", "ssd_square_pad" if cfg.get("crop", True) else "none"))
    variant = str(cfg.get("variant", "v2_res" if "s2_skip.0.weight" in state_dict else "v1"))

    model = KDEF_Stem2Ch_MS4_Symmetric(in_channels=in_channels, num_classes=num_classes, variant=variant)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    meta = {
        "model_name": str(cfg.get("model_name", "stem2ch_ms4_fuse")),
        "in_channels": in_channels,
        "mean": float(norm.get("mean", 0.407255)),
        "std": float(norm.get("std", 0.209657)),
        "use_crop": bool(cfg.get("crop", crop_mode != "none")),
        "crop_mode": crop_mode,
        "ssd_margin": float(cfg.get("ssd_margin", 1.25)),
        "use_clahe": bool(cfg.get("clahe", True)),
        "img_size": int(cfg.get("img_size", 160)),
        "epoch": ckpt.get("epoch"),
        "val_macro_f1": ckpt.get("val_macro_f1", ckpt.get("val_f1")),
    }
    return model, meta


def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số tham số có thể huấn luyện."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
