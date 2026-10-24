import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models.model import BaselineCNN
from src.data.dataset import KDEFDataset, get_transforms
from src.utils.device import get_device

def evaluate_model(data_dir: str, model_path: str):
    device = get_device()
    
    test_dataset = KDEFDataset(data_dir, split='test', transform=get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = BaselineCNN(num_classes=7)
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
    
    # Tính Accuracy
    all_preds_t = torch.tensor(all_preds)
    all_labels_t = torch.tensor(all_labels)
    acc = (all_preds_t == all_labels_t).float().mean().item()
    print(f"\nĐộ chính xác tổng thể (Accuracy) trên tập Test: {acc*100:.2f}%")
    
    # Tính Accuracy trên từng lớp
    print("\nChi tiết theo từng cảm xúc:")
    for i, name in enumerate(emotion_names):
        mask = (all_labels_t == i)
        if mask.sum() > 0:
            class_acc = (all_preds_t[mask] == all_labels_t[mask]).float().mean().item()
            print(f"- {name}: {class_acc*100:.2f}%")
    
    print("\nHoàn tất đánh giá!")

if __name__ == '__main__':
    kdef_path = r"F:\Hoc-May\data\balanced-kdef-dataset-by-dolly-prajapati-182"
    model_ckpt = "checkpoints/best_model.pth"
    evaluate_model(kdef_path, model_ckpt)

