from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io
from PIL import Image
from .inference import EmotionRecognizer
import os

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
