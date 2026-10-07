# Nhận Dạng Cảm Xúc Khuôn Mặt (Facial Emotion Recognition)

Dự án nhận dạng cảm xúc khuôn mặt đa văn hóa bằng mạng nơ-ron tích chập (CNN) trên tập dữ liệu KDEF.

## Cấu Trúc Mã Nguồn

- `src/data/`: Chứa các script tiền xử lý và EDA.
- `src/preprocessing/`: Các logic chuẩn hóa, cắt khuôn mặt (Face alignment, cropping).
- `src/models/`: Định nghĩa kiến trúc mạng (CNN, ResNet) và script huấn luyện.
- `src/utils/`: Các hàm hỗ trợ chung (logging, visualize).
- `src/api/`: Chứa API (FastAPI) và web interface (Streamlit).

## Cài Đặt Môi Trường

```bash
python -m venv venv
source venv/Scripts/activate # hoặc venv/bin/activate
pip install -r requirements.txt
```
