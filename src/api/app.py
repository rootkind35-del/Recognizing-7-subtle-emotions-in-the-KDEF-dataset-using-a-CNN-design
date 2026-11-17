import sys
import os

# Đảm bảo print() hỗ trợ tiếng Việt trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path để chạy script trực tiếp
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io
from PIL import Image
from src.api.inference import EMOTION_VI, EmotionRecognizer, resolve_default_checkpoint

app = FastAPI(
    title="Multicultural Emotion Recognition API",
    description="API nhận dạng 7 cảm xúc khuôn mặt trên bộ dữ liệu đa văn hóa (KDEF + RADIATE + FER-2013) sử dụng mô hình KDEF_Stem2Ch_MS4_Symmetric.",
    version="2.0.0",
)

# Khởi tạo recognizer khi app start (ưu tiên checkpoint tối ưu cốt lõi hoặc checkpoint notebook)
model_path = resolve_default_checkpoint()
recognizer = None

@app.on_event("startup")
def load_model():
    global recognizer
    if os.path.exists(model_path):
        recognizer = EmotionRecognizer(model_path)
    else:
        print(f"Cảnh báo: Không tìm thấy model tại {model_path}. Cần huấn luyện model trước.")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model": "KDEF_Stem2Ch_MS4_Symmetric",
        "dataset": "Multicultural (KDEF + RADIATE + FER-2013)",
        "checkpoint": model_path,
        "classes": recognizer.emotions if recognizer else [],
    }

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    if recognizer is None:
        return JSONResponse(status_code=503, content={"message": "Model chưa được tải hoặc chưa được huấn luyện."})
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(status_code=400, content={"message": "Tệp hình ảnh không hợp lệ."})

        pred_res = recognizer.predict_image(img, return_probabilities=True)
        if pred_res[0] is None:
            return JSONResponse(status_code=400, content={"message": "Không tìm thấy khuôn mặt trong ảnh."})
            
        emotion, conf, probs = pred_res
        return {
            "emotion": emotion,
            "emotion_vi": EMOTION_VI.get(emotion, emotion),
            "confidence": float(conf),
            "probabilities": probs,
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

