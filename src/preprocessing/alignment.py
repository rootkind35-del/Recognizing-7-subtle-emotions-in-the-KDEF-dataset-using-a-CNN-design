import cv2
import numpy as np
import os
from typing import Tuple, Optional

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

