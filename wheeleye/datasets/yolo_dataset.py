import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import glob

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(512, 512), transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transforms = transforms
        
        # Get all images
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.img_paths = []
        for ext in valid_exts:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, f'*{ext}')))
            
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Resize image
        img = img.resize(self.img_size)
        
        if self.transforms is not None:
            img_tensor = self.transforms(img)
        else:
            img_tensor = T.ToTensor()(img)
            
        # Get label path
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, f'{base_name}.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        # YOLO format: normalized cx, cy, w, h
                        cx, cy, w, h = map(float, parts[1:])
                        
                        # Scale to current img_size
                        cx = cx * self.img_size[0]
                        cy = cy * self.img_size[1]
                        w = w * self.img_size[0]
                        h = h * self.img_size[1]
                        
                        boxes.append([cx, cy, w, h])
                        labels.append(cls_id)
                        
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            
        return img_tensor, boxes, labels

def collate_fn(batch):
    """
    Custom collate_fn since boxes and labels have variable lengths per image.
    """
    imgs = []
    boxes = []
    labels = []
    
    for item in batch:
        imgs.append(item[0])
        boxes.append(item[1])
        labels.append(item[2])
        
    imgs = torch.stack(imgs, dim=0)
    return imgs, boxes, labels
