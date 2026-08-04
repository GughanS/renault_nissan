import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import sys
import numpy as np

# Adjust these imports depending on where you run this script from
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.datasets.classify_dataset import WheelClassifyDataset
from wheeleye.models.classifier import WheelEyeClassifier

def print_confusion_matrix(matrix, labels, title):
    print(f"\n--- {title} Confusion Matrix ---")
    
    # Calculate column widths
    max_label_len = max([len(str(l)) for l in labels] + [len("True \\ Pred")])
    col_width = max(max_label_len, 6)
    
    # Print Header
    header = "{:<{}} | ".format('True \\ Pred', col_width) + " | ".join([f"{str(l):<{col_width}}" for l in labels])
    print(header)
    print("-" * len(header))
    
    # Print Rows
    for i, true_label in enumerate(labels):
        row = f"{str(true_label):<{col_width}} | "
        row += " | ".join([f"{matrix[i, j]:<{col_width}}" for j in range(len(labels))])
        print(row)
        
    # Print Accuracy
    total = np.sum(matrix)
    correct = np.trace(matrix)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate WheelEye-Classify and generate Confusion Matrix")
    parser.add_argument('--img-dir', type=str, default='./data/crops', help="Path to cropped images")
    parser.add_argument('--csv-path', type=str, default='./data/crops_labels.csv', help="Path to labels CSV")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-path', type=str, default='./weights/classify_best.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.weight_path):
        print(f"Error: Weights not found at {args.weight_path}. Have you trained the model?")
        sys.exit(1)

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = WheelClassifyDataset(args.img_dir, args.csv_path, img_size=(224, 224), transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    model = WheelEyeClassifier().to(device)
    
    # Load weights
    ckpt = torch.load(args.weight_path, map_location=device, weights_only=True)
    # The saved weight might be the state_dict itself or a dictionary containing 'model_state_dict'
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()

    # Define Classes
    material_classes = ['Steel', 'Alloy']
    tier_classes = ['Standard', 'Premium', 'Luxury']
    size_classes = ['17_inch', '18_inch', '19_inch']
    
    # Initialize Confusion Matrices: rows are true, columns are predicted
    mat_cm = np.zeros((len(material_classes), len(material_classes)), dtype=int)
    tier_cm = np.zeros((len(tier_classes), len(tier_classes)), dtype=int)
    size_cm = np.zeros((len(size_classes), len(size_classes)), dtype=int)

    pbar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for imgs, target_mat, target_tier, target_size in pbar:
            imgs = imgs.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                out_mat, out_tier, out_size = model(imgs)
            
            pred_mat = torch.argmax(out_mat, dim=1).cpu().numpy()
            pred_tier = torch.argmax(out_tier, dim=1).cpu().numpy()
            pred_size = torch.argmax(out_size, dim=1).cpu().numpy()
            
            target_mat = target_mat.numpy()
            target_tier = target_tier.numpy()
            target_size = target_size.numpy()
            
            # Update confusion matrices
            for i in range(len(target_mat)):
                mat_cm[target_mat[i], pred_mat[i]] += 1
                tier_cm[target_tier[i], pred_tier[i]] += 1
                size_cm[target_size[i], pred_size[i]] += 1

    print_confusion_matrix(mat_cm, material_classes, "Material")
    print_confusion_matrix(tier_cm, tier_classes, "Tier")
    print_confusion_matrix(size_cm, size_classes, "Size")

if __name__ == '__main__':
    main()
