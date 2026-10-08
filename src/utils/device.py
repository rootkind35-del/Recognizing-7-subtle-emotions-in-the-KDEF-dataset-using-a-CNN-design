import torch

def get_device():
    """
    Kiểm tra và trả về thiết bị tính toán (CUDA/GPU hoặc CPU).
    In ra thông báo rõ ràng về thiết bị đang được sử dụng.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Phát hiện GPU: {gpu_name}. Dự án sẽ chạy bằng CUDA Toolkit.")
    else:
        device = torch.device("cpu")
        print("[INFO] Không tìm thấy GPU hợp lệ hoặc chưa cài CUDA Toolkit. Dự án sẽ chạy bằng CPU dự phòng.")
        
    return device
