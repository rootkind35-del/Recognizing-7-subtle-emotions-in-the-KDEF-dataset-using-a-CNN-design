import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import build_transfer_learning_model
from ..data.dataset import KDEFDataset, get_transforms
import copy

def train_model(data_dir: str, num_epochs: int = 20, batch_size: int = 32, learning_rate: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")
    
    # 1. Khởi tạo Dataset và DataLoader
    train_dataset = KDEFDataset(data_dir, split='train', transform=get_transforms('train'))
    val_dataset = KDEFDataset(data_dir, split='val', transform=get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 2. Xây dựng Mô hình
    model = build_transfer_learning_model('resnet18', num_classes=7, freeze_features=False)
    model = model.to(device)
    
    # 3. Hàm Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    early_stop_patience = 7
    epochs_no_improve = 0
    
    # 4. Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == 'val':
                scheduler.step(epoch_loss)
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
        if epochs_no_improve >= early_stop_patience:
            print("Dừng sớm do validation loss không cải thiện.")
            break
            
    print(f"Huấn luyện hoàn tất. Best Val Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    
    # Lưu mô hình
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    print("Đã lưu best checkpoint tại checkpoints/best_model.pth")
    return model

if __name__ == '__main__':
    train_model(data_dir=r"F:\Hoc-May\data\balanced-kdef-dataset-by-dolly-prajapati-182", num_epochs=20)
