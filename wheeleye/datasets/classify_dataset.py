import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class WheelClassifyDataset(Dataset):
    def __init__(self, img_dir, csv_path, img_size=(224, 224), transforms=None):
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    'filename': row['filename'],
                    'material': int(row['material']),
                    'tier': int(row['tier']),
                    'size': int(row['size'])
                })
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample['filename'])
        
        img = Image.open(img_path).convert('RGB')
        
        # Resize image for the classifier backbone
        img = img.resize(self.img_size)
        
        if self.transforms is not None:
            img_tensor = self.transforms(img)
        else:
            img_tensor = T.ToTensor()(img)
            
        target_material = torch.tensor(sample['material'], dtype=torch.long)
        target_tier = torch.tensor(sample['tier'], dtype=torch.long)
        target_size = torch.tensor(sample['size'], dtype=torch.long)
            
        return img_tensor, target_material, target_tier, target_size
