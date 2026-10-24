import torch
import torch.nn as nn
from torchvision import models

class BaselineCNN(nn.Module):
    """
    Mạng CNN cơ bản được xây dựng từ đầu (từ Task 3.1).
    Sử dụng để kiểm tra pipeline.
    """
    def __init__(self, num_classes: int = 7):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Ảnh đầu vào 224x224, qua 3 lần MaxPool (chia 8) -> 28x28
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_transfer_learning_model(model_name: str = 'resnet18', num_classes: int = 7, freeze_features: bool = True) -> nn.Module:
    """
    Tải mô hình Pre-trained và tùy biến cho bài toán nhận dạng cảm xúc.
    
    Args:
        model_name (str): Tên mô hình ('resnet18' hoặc 'efficientnet_b0').
        num_classes (int): Số lượng lớp phân loại (7 cho KDEF).
        freeze_features (bool): Đóng băng các lớp feature extraction nếu True.
    """
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        
        if freeze_features:
            for param in model.parameters():
                param.requires_grad = False
                
        # Thay thế lớp Fully Connected cuối cùng
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        
        if freeze_features:
            for param in model.parameters():
                param.requires_grad = False
                
        # Thay thế lớp phân loại
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Chưa hỗ trợ mô hình {model_name}.")
        
    return model

