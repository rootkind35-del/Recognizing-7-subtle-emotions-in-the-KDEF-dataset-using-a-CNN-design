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
from src.api.inference import EmotionRecognizer

app = FastAPI(title="Emotion Recognition API", description="API nhận dạng cảm xúc khuôn mặt dựa trên KDEF dataset.")

# Khởi tạo recognizer khi app start
model_path = "checkpoints/best_model.pth"
recognizer = None

@app.on_event("startup")
def load_model():
    global recognizer
    if os.path.exists(model_path):
        recognizer = EmotionRecognizer(model_path)
    else:
        print(f"Cảnh báo: Không tìm thấy model tại {model_path}. Cần huấn luyện model trước.")

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    if recognizer is None:
        return JSONResponse(status_code=503, content={"message": "Model chưa được tải hoặc chưa được huấn luyện."})
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        emotion, conf = recognizer.predict_image(img)
        
        if emotion is None:
            return JSONResponse(status_code=400, content={"message": "Không tìm thấy khuôn mặt trong ảnh."})
            
        return {"emotion": emotion, "confidence": float(conf)}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

