import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class FaceAligner:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        # Khởi tạo model MediaPipe với độ tin cậy 0.5
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )

    def align_and_crop(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Nhận diện và cắt vùng khuôn mặt từ ảnh gốc.
        """
        # MediaPipe yêu cầu ảnh RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)

        if not results.detections:
            return None

        # Lấy khuôn mặt đầu tiên (giả định mỗi ảnh chân dung chỉ có 1 người chính)
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape

        # Chuyển đổi tọa độ tương đối sang pixel
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        # Mở rộng bounding box một chút để lấy trọn vẹn khuôn mặt
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(iw, x + w + margin_x)
        y2 = min(ih, y + h + margin_y)

        cropped_face = image[y1:y2, x1:x2]
        
        # Resize về kích thước chuẩn
        resized_face = cv2.resize(cropped_face, target_size)
        return resized_face
