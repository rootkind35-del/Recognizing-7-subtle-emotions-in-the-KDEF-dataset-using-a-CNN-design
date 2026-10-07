import cv2
import torch
import numpy as np
from PIL import Image
from ..models.model import build_transfer_learning_model
from ..preprocessing.alignment import FaceAligner
from ..data.dataset import get_transforms

class EmotionRecognizer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_transfer_learning_model('resnet18', num_classes=7)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.aligner = FaceAligner()
        self.transform = get_transforms('test')
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def predict_image(self, image: np.ndarray):
        # 1. Phát hiện và cắt khuôn mặt
        face = self.aligner.align_and_crop(image)
        if face is None:
            return None, "No face detected"
            
        # 2. Tiền xử lý
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        # 3. Suy luận
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
            
        return self.emotions[predicted.item()], confidence.item()

def run_webcam(model_path: str):
    recognizer = EmotionRecognizer(model_path)
    cap = cv2.VideoCapture(0)
    
    print("Nhấn 'q' để thoát.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        emotion, conf = recognizer.predict_image(frame)
        
        if emotion:
            text = f"{emotion}: {conf:.2f}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_webcam("checkpoints/best_model.pth")
