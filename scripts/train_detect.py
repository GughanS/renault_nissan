import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add repo root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wheeleye.models.detector import WheelEyeDetector
from wheeleye.datasets.yolo_dataset import YOLODataset, collate_fn
from wheeleye.utils.anchors import generate_anchors, assign_targets, encode_boxes
from wheeleye.utils.loss import DetectionLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train WheelEye-Detect")
    parser.add_argument('--img-dir', type=str, required=True)
    parser.add_argument('--label-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weights-dir', type=str, default='weights')
    parser.add_argument('--resume', action='store_true', help='Resume from last.pt if exists')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def train(args):
    device = torch.device(args.device)
    os.makedirs(args.weights_dir, exist_ok=True)
    
    # Dataset and DataLoader
    dataset = YOLODataset(args.img_dir, args.label_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    
    # Model
    model = WheelEyeDetector(num_classes=4, num_anchors=9).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss Function
    criterion = DetectionLoss(num_classes=4)
    
    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    # Anchors
    # Strides corresponding to C3, C4, C5 in our FPN
    strides = [8, 16, 32]
    base_sizes = [32, 64, 128]
    scales = [1, 2**(1/3), 2**(2/3)]
    aspect_ratios = [0.5, 1.0, 2.0]
    
    anchors = generate_anchors((512, 512), strides, base_sizes, scales, aspect_ratios)
    anchors = anchors.to(device)
    
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume
    last_pt = os.path.join(args.weights_dir, 'last.pt')
    if args.resume and os.path.exists(last_pt):
        checkpoint = torch.load(last_pt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch_idx, (imgs, boxes, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            B = imgs.shape[0]
            
            # Prepare targets for the batch
            target_classes = []
            target_boxes = []
            
            for b in range(B):
                gt_b = boxes[b].to(device)
                gt_l = labels[b].to(device)
                t_cls, t_box = assign_targets(anchors, gt_b, gt_l)
                target_classes.append(t_cls)
                target_boxes.append(t_box)
                
            target_classes = torch.stack(target_classes, dim=0)
            target_boxes = torch.stack(target_boxes, dim=0)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                pred_cls, pred_reg = model(imgs)
                cls_loss, reg_loss, loss = criterion(pred_cls, pred_reg, target_classes, target_boxes, anchors)
            
            if loss > 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_cls_loss += cls_loss.item()
                epoch_reg_loss += reg_loss.item()
                epoch_total_loss += loss.item()
                
            print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
            
        avg_loss = epoch_total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
            
        torch.save(checkpoint, last_pt)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(args.weights_dir, 'best.pt'))
            print("Saved new best model.")

if __name__ == '__main__':
    args = parse_args()
    train(args)
