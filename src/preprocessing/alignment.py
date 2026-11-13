import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, Union

# Aspect-aware studio portrait crop box [x1, y1, x2, y2] relative to image dimensions
KDEF_STUDIO_CROP_BOX = (0.04, 0.16, 0.96, 0.86)


def preprocess_kdef_image(
    img_or_path: Union[str, Path, np.ndarray],
    use_aspect_crop: bool = True,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_blend: float = 0.8,
    img_size: int = 128
) -> np.ndarray:
    """
    Standardized KDEF Image Preprocessing Pipeline:
      1. Grayscale conversion.
      2. Aspect-Aware Studio Face Crop [0.16H:0.86H, 0.04W:0.96W] eliminating
         35.65% non-facial pixels (shoulders, hair crown) and reducing squash
         distortion from 26.25% down to 3.08%.
      3. cv2.resize to (img_size, img_size) with cv2.INTER_AREA (anti-aliased downsampling).
      4. CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=(8, 8)) blended
         at 0.8 * enhanced + 0.2 * raw.
    Returns uint8 array of shape (img_size, img_size) in [0, 255].
    """
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_or_path}")
    elif isinstance(img_or_path, np.ndarray):
        if img_or_path.ndim == 3:
            img = cv2.cvtColor(img_or_path, cv2.COLOR_BGR2GRAY)
        elif img_or_path.ndim == 2:
            img = img_or_path.copy()
        else:
            raise ValueError(f"Unexpected image array shape: {img_or_path.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(img_or_path)}")

    h, w = img.shape[:2]
    if use_aspect_crop:
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = KDEF_STUDIO_CROP_BOX
        y1, y2 = int(round(y1_ratio * h)), int(round(y2_ratio * h))
        x1, x2 = int(round(x1_ratio * w)), int(round(x2_ratio * w))
        img = img[y1:y2, x1:x2]

    # Resize with cv2.INTER_AREA for clean downsampling
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        if clahe_blend < 1.0:
            img = cv2.addWeighted(enhanced, clahe_blend, img, 1.0 - clahe_blend, 0)
        else:
            img = enhanced

    return img.astype(np.uint8)


class FaceAligner:
    def __init__(self):
        # Sử dụng Haar Cascade (Thuật toán Học Máy dựa trên AdaBoost)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def align_and_crop(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Nhận diện và cắt vùng khuôn mặt bằng Haar Cascade.
        Đã tối ưu tham số để nhạy hơn với các khuôn mặt ở góc nghiêng nhẹ.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Cải thiện độ tương phản (Histogram Equalization) giúp Haar Cascade bắt tốt hơn
        gray = cv2.equalizeHist(gray)
        
        # Nhận diện khuôn mặt (giảm scaleFactor và minNeighbors để tăng độ nhạy)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Lấy khuôn mặt to nhất
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]

        ih, iw, _ = image.shape

        # Mở rộng bounding box
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(iw, x + w + margin_x)
        y2 = min(ih, y + h + margin_y)

        cropped_face = image[y1:y2, x1:x2]
        
        if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
            return None

        # Resize về kích thước chuẩn
        resized_face = cv2.resize(cropped_face, target_size)
        return resized_face

