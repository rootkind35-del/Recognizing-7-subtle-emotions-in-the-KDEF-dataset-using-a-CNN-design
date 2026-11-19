# -*- coding: utf-8 -*-
"""Streamlit Web UI for Multicultural Facial Emotion Recognition.

Implements the interface described in Section 4.4 of the project report:
  - Tab 1: "Tải ảnh lên" (Image upload, face bounding box, 7 Vietnamese probability bars)
  - Tab 2: "Camera trực tiếp" (Live webcam capture & inference)
  - Sidebar: Model architecture details, multicultural training sources, and locked-test metrics.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Setup project root in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.inference import EMOTION_VI, EmotionRecognizer, resolve_default_checkpoint  # noqa: E402
from src.preprocessing.alignment import detect_face_ssd  # noqa: E402

st.set_page_config(
    page_title="Nhận Diện Cảm Xúc Khuôn Mặt",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMOTION_ICONS = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲",
}


@st.cache_resource
def load_recognizer() -> Optional[EmotionRecognizer]:
    ckpt_path = resolve_default_checkpoint()
    if os.path.exists(ckpt_path):
        return EmotionRecognizer(ckpt_path)
    return None


def draw_face_box(img_bgr: np.ndarray, aligner) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Draws a green bounding box around the detected face."""
    annotated = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    box = None

    # Try SSD first
    ssd_box, _ = detect_face_ssd(img_bgr, min_conf=0.20)
    if ssd_box is not None:
        x1n, y1n, x2n, y2n = ssd_box
        ix1, iy1 = int(round(x1n * w)), int(round(y1n * h))
        ix2, iy2 = int(round(x2n * w)), int(round(y2n * h))
        box = (ix1, iy1, ix2, iy2)
    elif aligner.face_cascade is not None:
        gray = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        faces = aligner.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            x, y, fw, fh = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
            box = (x, y, x + fw, y + fh)

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), max(2, int(round(min(h, w) / 200))))
    return annotated, box


def render_probability_bars(probs_dict: Dict[str, float], predicted: str) -> None:
    """Renders 7 Vietnamese emotion probability progress bars."""
    st.markdown("### Xác suất 7 cảm xúc")
    for emo, prob in probs_dict.items():
        vi_name = EMOTION_VI.get(emo, emo)
        icon = EMOTION_ICONS.get(emo, "•")
        pct = prob * 100.0
        is_pred = (emo == predicted)
        prefix = "**" if is_pred else ""
        suffix = "** (dự đoán)" if is_pred else ""
        col1, col2 = st.columns([3, 7])
        with col1:
            st.markdown(f"{prefix}{icon} {vi_name} ({emo}){suffix}")
        with col2:
            st.progress(float(np.clip(prob, 0.0, 1.0)), text=f"{pct:.1f}%")


def main() -> None:
    # -----------------------------------------------------------------------
    # Sidebar: Model, Data & Benchmark Information
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Về mô hình")
        st.markdown(
            """
            **KDEF_Stem2Ch_MS4_Symmetric (v2_res)**
            - Mạng nơ-ron tích chập 4 khối đa tỉ lệ đối xứng + Residual Skips (1,35 triệu tham số, ~5,19 MB).
            - Ảnh đầu vào: Xám 160×160 + CLAHE.
            - Tốc độ suy luận CPU: ~17,3 ms/ảnh (~58 FPS).
            """
        )
        st.markdown("---")
        st.header("Dữ liệu đa văn hóa")
        st.markdown(
            """
            - **41.269 ảnh** từ 3 nguồn:
              - **KDEF** (70 người Thụy Điển, 5 góc).
              - **RADIATE** (109 người Mỹ: Phi, Á, Latinh, Âu).
              - **FER-2013** (35.887 ảnh ngoài đời thực).
            - Phân chia theo người (**Subject-Disjoint**).
            - Không trùng lặp MD5 chéo tập (0% rò rỉ).
            """
        )
        st.markdown("---")
        st.header("Hiệu năng trên tập kiểm tra")
        st.markdown(
            """
            *Tập test độc lập 4.875 ảnh (người chưa từng gặp):*
            - **Accuracy:** **73,39%** (KDEF 86,5%, RADIATE 85,0%, FER 68,8%)
            - **Macro-F1:** **72,82%** (Vượt HOG+SVM +31,2% với kích thước nhẹ hơn ResNet-18 ~8,3×)
            - **ROC-AUC (OvR macro):** **0,9450**
            """
        )
        st.info("💡 **Mẹo:** Dùng ảnh chụp góc gần chính diện, khuôn mặt rõ nét và đủ sáng để đạt kết quả tốt nhất.")

    # -----------------------------------------------------------------------
    # Main Header
    # -----------------------------------------------------------------------
    st.title("🎭 Nhận Diện Cảm Xúc Khuôn Mặt Đa Văn Hóa")
    st.write(
        "Ứng dụng học máy nhận diện một trong 7 cảm xúc cơ bản từ ảnh khuôn mặt thuộc nhiều nhóm người và nguồn gốc."
    )

    recognizer = load_recognizer()
    if recognizer is None:
        st.error(
            "Không tìm thấy mô hình tại `checkpoints/final_model.pt`. "
            "Vui lòng đảm bảo checkpoint đã được lưu."
        )
        return

    # -----------------------------------------------------------------------
    # Tabs: Upload vs Live Camera
    # -----------------------------------------------------------------------
    tab_upload, tab_camera = st.tabs(["📁 Tải ảnh lên", "📹 Camera trực tiếp"])

    with tab_upload:
        st.write("Tải lên một bức ảnh chân dung (JPG, JPEG, PNG) để nhận diện cảm xúc:")
        uploaded_file = st.file_uploader(
            "Chọn tệp ảnh...",
            type=["jpg", "jpeg", "png"],
            help="Tải ảnh khuôn mặt gần chính diện",
        )

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            with st.spinner("Đang phân tích khuôn mặt và cảm xúc..."):
                annotated_bgr, box = draw_face_box(img_bgr, recognizer.aligner)
                pred_emo, conf, probs_dict = recognizer.predict_image(img_bgr, return_probabilities=True)

            col_left, col_right = st.columns([5, 5])
            with col_left:
                st.image(
                    cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                    caption="Ảnh đã tải lên (khung xanh là vùng mặt nhận diện)",
                    use_container_width=True,
                )

            with col_right:
                if pred_emo is not None:
                    vi_emo = EMOTION_VI.get(pred_emo, pred_emo)
                    icon = EMOTION_ICONS.get(pred_emo, "•")
                    st.success(f"### Dự đoán: {icon} {vi_emo} ({pred_emo}) — Độ tin cậy: {conf * 100:.1f}%")
                    render_probability_bars(probs_dict, pred_emo)
                else:
                    st.warning("⚠️ Không phát hiện được khuôn mặt hợp lệ trong ảnh.")

    with tab_camera:
        st.write("Chụp ảnh từ webcam để nhận diện cảm xúc tức thì:")
        camera_img = st.camera_input("Chụp ảnh từ webcam")

        if camera_img is not None:
            pil_cam = Image.open(camera_img).convert("RGB")
            cam_bgr = cv2.cvtColor(np.array(pil_cam), cv2.COLOR_RGB2BGR)

            with st.spinner("Đang xử lý khung hình camera..."):
                annotated_cam, box = draw_face_box(cam_bgr, recognizer.aligner)
                pred_emo, conf, probs_dict = recognizer.predict_image(cam_bgr, return_probabilities=True)

            col_c1, col_c2 = st.columns([5, 5])
            with col_c1:
                st.image(
                    cv2.cvtColor(annotated_cam, cv2.COLOR_BGR2RGB),
                    caption="Ảnh chụp từ camera",
                    use_container_width=True,
                )
            with col_c2:
                if pred_emo is not None:
                    vi_emo = EMOTION_VI.get(pred_emo, pred_emo)
                    icon = EMOTION_ICONS.get(pred_emo, "•")
                    st.success(f"### {icon} {vi_emo} ({pred_emo}) — {conf * 100:.1f}%")
                    render_probability_bars(probs_dict, pred_emo)
                else:
                    st.warning("⚠️ Vui lòng nhìn thẳng vào camera và đảm bảo đủ ánh sáng.")


if __name__ == "__main__":
    main()
