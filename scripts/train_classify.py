import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

# Adjust these imports depending on where you run this script from
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.datasets.classify_dataset import WheelClassifyDataset
from wheeleye.models.classifier import WheelEyeClassifier

def main():
    parser = argparse.ArgumentParser(description="Train WheelEye-Classify")
    parser.add_argument('--img-dir', type=str, default='./data/crops', help="Path to cropped images")
    parser.add_argument('--csv-path', type=str, default='./data/crops_labels.csv', help="Path to labels CSV")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-dir', type=str, default='./weights')
    args = parser.parse_args()

    os.makedirs(args.weight_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & Dataloader
    # In a real scenario we'd use stronger augmentations here. 
    # For now, minimal transforms for our synthetic data.
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = WheelClassifyDataset(args.img_dir, args.csv_path, img_size=(224, 224), transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    # Model
    model = WheelEyeClassifier().to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        # We use a progress bar for interactive viewing
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, target_mat, target_tier, target_size in pbar:
            imgs = imgs.to(device)
            target_mat = target_mat.to(device)
            target_tier = target_tier.to(device)
            target_size = target_size.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out_mat, out_tier, out_size = model(imgs)
                
                loss_mat = criterion(out_mat, target_mat)
                loss_tier = criterion(out_tier, target_tier)
                loss_size = criterion(out_size, target_size)
                
                loss = loss_mat + loss_tier + loss_size
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        # Checkpointing
        last_ckpt = os.path.join(args.weight_dir, 'classify_last.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, last_ckpt)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(args.weight_dir, 'classify_best.pt')
            torch.save(model.state_dict(), best_ckpt)
            print(f"--> Saved new best model with loss {best_loss:.4f}")

if __name__ == '__main__':
    main()
