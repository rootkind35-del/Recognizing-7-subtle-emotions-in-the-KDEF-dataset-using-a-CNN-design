import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class KDEFDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        """
        Khởi tạo tập dữ liệu KDEF.
        
        Args:
            data_dir (str): Đường dẫn đến thư mục root của dữ liệu (chứa train/val/test).
            split (str): 'train', 'val', hoặc 'test'.
            transform: Các phép biến đổi PyTorch (ví dụ: data augmentation).
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        self.emotion_to_idx = {
            'Angry': 0,
            'Disgust': 1,
            'Fear': 2,
            'Happy': 3,
            'Neutral': 4,
            'Sad': 5,
            'Surprise': 6
        }
        
        self._load_dataset()

    def _load_dataset(self):
        split_dir = os.path.join(self.data_dir, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {split_dir}")
            
        for emotion, idx in self.emotion_to_idx.items():
            emotion_dir = os.path.join(split_dir, emotion)
            if not os.path.exists(emotion_dir):
                continue
                
            img_paths = glob.glob(os.path.join(emotion_dir, "*.jpg"))
            self.image_paths.extend(img_paths)
            self.labels.extend([idx] * len(img_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Mở ảnh bằng PIL để tương thích với torchvision.transforms
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(split: str = 'train', img_size: int = 224):
    """
    Trả về cấu hình augmentation tương ứng cho từng tập.
    """
    # Chuẩn hóa theo ImageNet mean & std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Tập validation và test không dùng random augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

