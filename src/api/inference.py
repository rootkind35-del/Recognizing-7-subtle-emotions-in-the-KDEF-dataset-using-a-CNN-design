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

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models.model import load_kdef_checkpoint
from src.preprocessing.alignment import (
    DEFAULT_CROP_MODE,
    DEFAULT_SSD_MARGIN,
    FaceAligner,
    preprocess_kdef_image,
)
from src.utils.device import get_device

DEFAULT_CHECKPOINT_CANDIDATES = (
    "checkpoints/final_model.pt",
    "checkpoints/cnn_multicultural.pt",
)

EMOTION_VI = {
    "Angry": "Giận dữ",
    "Disgust": "Ghê tởm",
    "Fear": "Sợ hãi",
    "Happy": "Vui vẻ",
    "Neutral": "Trung tính",
    "Sad": "Buồn",
    "Surprise": "Ngạc nhiên",
}


def resolve_default_checkpoint() -> str:
    """Trả về đường dẫn checkpoint chuẩn duy nhất (checkpoints/final_model.pt)."""
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        full_path = os.path.join(project_root, candidate)
        if os.path.exists(full_path):
            return full_path
        if os.path.exists(candidate):
            return candidate
    return DEFAULT_CHECKPOINT_CANDIDATES[0]


class EmotionRecognizer:
    def __init__(self, model_path: str = None, use_tta: bool = False):
        self.device = get_device()
        self.use_tta = use_tta
        resolved_path = model_path or resolve_default_checkpoint()
        self.model, self.meta = load_kdef_checkpoint(resolved_path, device=self.device, num_classes=7)

        self.img_size = self.meta["img_size"]
        self.mean = self.meta["mean"]
        self.std = self.meta["std"]
        self.use_crop = self.meta.get("use_crop", True)
        self.crop_mode = self.meta.get("crop_mode", DEFAULT_CROP_MODE)
        self.ssd_margin = float(self.meta.get("ssd_margin", DEFAULT_SSD_MARGIN))
        self.use_clahe = self.meta["use_clahe"]
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if self.use_clahe else None

        self.aligner = FaceAligner(
            use_ssd=self.crop_mode.startswith("ssd"),
            ssd_margin=self.ssd_margin,
        )
        self.normalize = transforms.Normalize(mean=[self.mean], std=[self.std])
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def _preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """Applies exact aspect/intensity preprocessing matching training pipeline."""
        if face_img.ndim == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        elif face_img.ndim == 2:
            gray = face_img.copy()
        else:
            raise ValueError(f"Invalid image dimensions: {face_img.shape}")

        if self.clahe is not None:
            eq = self.clahe.apply(gray)
            gray = cv2.addWeighted(eq, 0.8, gray, 0.2, 0)

        # Anti-aliased downsampling
        if gray.shape[:2] != (self.img_size, self.img_size):
            gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float() / 255.0
        tensor = self.normalize(tensor).to(self.device)
        return tensor

    def get_gradcam_target_layers(self):
        """Trả về tầng đích Grad-CAM an toàn cho KDEF_Stem2Ch_MS4_Symmetric."""
        if hasattr(self.model, "get_gradcam_target_layers"):
            return self.model.get_gradcam_target_layers()
        return [self.model.s4_ms.eca]

    def predict_image(
        self,
        image: np.ndarray,
        auto_fallback: bool = True,
        return_probabilities: bool = False,
    ):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            if return_probabilities:
                return None, 0.0, {}
            return None, 0.0

        # 1. Phát hiện và cắt khuôn mặt theo đúng ssd_pad_160_1ch (SSD square-pad + fallback)
        if not self.use_crop or self.crop_mode == "none":
            face = image
        else:
            face = self.aligner.align_and_crop(
                image,
                target_size=(self.img_size, self.img_size),
                margin=self.ssd_margin,
            )
            if face is None:
                if auto_fallback:
                    try:
                        face = preprocess_kdef_image(
                            image,
                            use_aspect_crop=self.use_crop,
                            use_clahe=False,  # clahe sẽ được áp dụng trong _preprocess_face
                            img_size=self.img_size,
                            crop_mode=self.crop_mode,
                            ssd_margin=self.ssd_margin,
                        )
                    except Exception:
                        if return_probabilities:
                            return None, 0.0, {}
                        return None, "No face detected"
                else:
                    if return_probabilities:
                        return None, 0.0, {}
                    return None, "No face detected"

        # 2. Tiền xử lý grayscale 160x160 (+ CLAHE nếu checkpoint sử dụng)
        input_tensor = self._preprocess_face(face)

        # 3. Suy luận (kèm Flip-TTA nếu bật)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            if self.use_tta:
                outputs = 0.5 * (outputs + self.model(torch.flip(input_tensor, dims=[3])))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)

        pred_emotion = self.emotions[predicted.item()]
        conf_val = float(confidence.item())
        probs_dict = {
            emo: float(probabilities[i].item()) for i, emo in enumerate(self.emotions)
        }

        if return_probabilities:
            return pred_emotion, conf_val, probs_dict
        return pred_emotion, conf_val

    def predict_image_full(self, image: np.ndarray):
        """Trả về đầy đủ: (emotion, confidence, probabilities_dict, face_crop)."""
        return self.predict_image(image, auto_fallback=True, return_probabilities=True)

def run_webcam(model_path: str = None):
    recognizer = EmotionRecognizer(model_path)
    cap = cv2.VideoCapture(0)
    
    print("Nhấn 'q' để thoát.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        emotion, conf = recognizer.predict_image(frame)
        
        if emotion:
            text = f"{emotion}: {conf*100:.1f}%"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_webcam()

