import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import glob
import numpy as np


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(512, 512), transforms=None,
                 mosaic=False, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transforms = transforms
        self.mosaic = mosaic
        self.augment = augment

        # Get all images
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.img_paths = []
        for ext in valid_exts:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, f'*{ext}')))

    def __len__(self):
        return len(self.img_paths)

    def _load_image_and_labels(self, idx):
        """Load a single image and its YOLO-format labels."""
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # Resize image
        img = img.resize(self.img_size)

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

                        # Scale to current img_size (pixel coordinates)
                        cx = cx * self.img_size[0]
                        cy = cy * self.img_size[1]
                        w = w * self.img_size[0]
                        h = h * self.img_size[1]

                        boxes.append([cx, cy, w, h])
                        labels.append(cls_id)

        return img, boxes, labels

    def _apply_augmentations(self, img, boxes, labels):
        """Apply standard augmentations: ColorJitter, HFlip, HSV shifts."""
        if not self.augment:
            return img, boxes, labels

        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            img = TF.hflip(img)
            new_boxes = []
            for cx, cy, w, h in boxes:
                cx = self.img_size[0] - cx
                new_boxes.append([cx, cy, w, h])
            boxes = new_boxes

        # Color jitter
        if random.random() > 0.5:
            jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            img = jitter(img)

        # Random HSV shift
        if random.random() > 0.5:
            img_array = np.array(img).astype(np.float32)
            # Brightness shift
            img_array *= random.uniform(0.8, 1.2)
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

        return img, boxes, labels

    def _mosaic_getitem(self, idx):
        """
        Mosaic augmentation: composite 4 images into a 2x2 grid.
        Heavily improves detection of small objects like fasteners.
        """
        img_w, img_h = self.img_size

        # Random split point for the 4 quadrants
        split_x = random.randint(int(img_w * 0.25), int(img_w * 0.75))
        split_y = random.randint(int(img_h * 0.25), int(img_h * 0.75))

        # Sample 3 additional random indices
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]

        # Canvas
        canvas = Image.new('RGB', (img_w, img_h), (114, 114, 114))

        all_boxes = []
        all_labels = []

        # Quadrant definitions: (paste_x, paste_y, crop_region_from_source)
        # Each image is resized to img_size, then we crop a region to fit the quadrant
        for i, img_idx in enumerate(indices):
            img, boxes, labels = self._load_image_and_labels(img_idx)

            if i == 0:  # Top-left
                # Place the bottom-right portion of the image into top-left quadrant
                crop_x1 = img_w - split_x
                crop_y1 = img_h - split_y
                paste_x, paste_y = 0, 0
                quad_w, quad_h = split_x, split_y
            elif i == 1:  # Top-right
                crop_x1 = 0
                crop_y1 = img_h - split_y
                paste_x, paste_y = split_x, 0
                quad_w, quad_h = img_w - split_x, split_y
            elif i == 2:  # Bottom-left
                crop_x1 = img_w - split_x
                crop_y1 = 0
                paste_x, paste_y = 0, split_y
                quad_w, quad_h = split_x, img_h - split_y
            else:  # Bottom-right
                crop_x1 = 0
                crop_y1 = 0
                paste_x, paste_y = split_x, split_y
                quad_w, quad_h = img_w - split_x, img_h - split_y

            # Crop region from source image
            crop_region = img.crop((crop_x1, crop_y1, crop_x1 + quad_w, crop_y1 + quad_h))
            canvas.paste(crop_region, (paste_x, paste_y))

            # Adjust bounding boxes
            for box, lbl in zip(boxes, labels):
                cx, cy, w, h = box

                # Shift box coordinates: from source image space to canvas space
                new_cx = cx - crop_x1 + paste_x
                new_cy = cy - crop_y1 + paste_y

                # Clip box to quadrant boundaries
                x1 = max(paste_x, new_cx - w / 2)
                y1 = max(paste_y, new_cy - h / 2)
                x2 = min(paste_x + quad_w, new_cx + w / 2)
                y2 = min(paste_y + quad_h, new_cy + h / 2)

                clipped_w = x2 - x1
                clipped_h = y2 - y1

                # Discard boxes that become too small (< 4px in either dimension)
                if clipped_w < 4 or clipped_h < 4:
                    continue

                clipped_cx = (x1 + x2) / 2
                clipped_cy = (y1 + y2) / 2

                all_boxes.append([clipped_cx, clipped_cy, clipped_w, clipped_h])
                all_labels.append(lbl)

        return canvas, all_boxes, all_labels

    def __getitem__(self, idx):
        if self.mosaic and random.random() > 0.3:
            # 70% chance of mosaic when enabled
            img, boxes, labels = self._mosaic_getitem(idx)
        else:
            img, boxes, labels = self._load_image_and_labels(idx)

        # Apply augmentations
        img, boxes, labels = self._apply_augmentations(img, boxes, labels)

        # Convert to tensor
        if self.transforms is not None:
            img_tensor = self.transforms(img)
        else:
            img_tensor = T.ToTensor()(img)

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
