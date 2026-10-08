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

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.model import build_transfer_learning_model
from src.data.dataset import KDEFDataset, get_transforms

def evaluate_model(data_dir: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = KDEFDataset(data_dir, split='test', transform=get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = build_transfer_learning_model('resnet18', num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    print("Báo cáo phân loại (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=emotion_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_names, yticklabels=emotion_names, cmap='Blues')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.savefig('confusion_matrix.png')
    print("Đã lưu ma trận nhầm lẫn tại confusion_matrix.png")

if __name__ == '__main__':
    kdef_path = r"F:\Hoc-May\data\balanced-kdef-dataset-by-dolly-prajapati-182"
    model_ckpt = "checkpoints/best_model.pth"
    evaluate_model(kdef_path, model_ckpt)

