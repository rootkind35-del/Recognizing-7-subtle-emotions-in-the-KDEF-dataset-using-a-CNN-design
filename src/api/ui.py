import sys
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Thêm thư mục gốc vào sys.path để chạy script trực tiếp
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api.inference import EmotionRecognizer

st.set_page_config(page_title="Nhận Dạng Cảm Xúc", layout="centered")

@st.cache_resource
def load_model():
    model_path = "checkpoints/best_model.pth"
    if os.path.exists(model_path):
        return EmotionRecognizer(model_path)
    return None

st.title("🎭 Nhận Dạng Cảm Xúc Khuôn Mặt")
st.write("Tải lên một bức ảnh có khuôn mặt để dự đoán cảm xúc.")

recognizer = load_model()

if recognizer is None:
    st.error("Không tìm thấy mô hình tại `checkpoints/best_model.pth`. Vui lòng huấn luyện mô hình trước.")
else:
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Đọc ảnh từ file upload
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        
        # Chuyển đổi sang format OpenCV (BGR)
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3: # RGB
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4: # RGBA
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_cv = img_array
            
        with st.spinner("Đang phân tích..."):
            emotion, conf = recognizer.predict_image(img_cv)
            
        if emotion:
            st.success(f"**Cảm xúc dự đoán:** {emotion} ({conf:.2f}%)")
        else:
            st.warning("Không phát hiện được khuôn mặt hợp lệ trong ảnh.")

