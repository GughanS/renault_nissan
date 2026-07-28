import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add repo root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wheeleye.models.detector import WheelEyeDetector
from wheeleye.datasets.yolo_dataset import YOLODataset, collate_fn
from wheeleye.utils.anchors import generate_anchors, task_aligned_assign, assign_targets, decode_boxes
from wheeleye.utils.loss import DetectionLoss
from wheeleye.utils.ema import ModelEMA

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
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Number of linear warmup epochs before cosine annealing')
    parser.add_argument('--eta-min', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--use-tal', action='store_true', default=True,
                        help='Use Task-Aligned Assigner (default: True)')
    parser.add_argument('--mosaic', action='store_true', default=True,
                        help='Enable Mosaic augmentation (default: True)')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable standard augmentations (default: True)')
    return parser.parse_args()

def train(args):
    device = torch.device(args.device)
    os.makedirs(args.weights_dir, exist_ok=True)
    
    # Dataset and DataLoader — with Mosaic and augmentation support
    mosaic = getattr(args, 'mosaic', False)
    augment = getattr(args, 'augment', False)
    dataset = YOLODataset(args.img_dir, args.label_dir,
                          mosaic=mosaic, augment=augment)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)
    
    # Model
    model = WheelEyeDetector(num_classes=4, num_anchors=9).to(device)
    
    # Multi-GPU support via DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs for training via DataParallel!")
        model = torch.nn.DataParallel(model)
        
    # Exponential Moving Average
    ema = ModelEMA(model)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning Rate Schedulers: Linear Warmup → Cosine Annealing
    warmup_epochs = getattr(args, 'warmup_epochs', 3)
    eta_min = getattr(args, 'eta_min', 1e-6)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=eta_min / args.lr,
                                end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer,
                                         T_max=args.epochs - warmup_epochs,
                                         eta_min=eta_min)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])
    
    # Loss Function
    criterion = DetectionLoss(num_classes=4)
    
    # AMP Scaler
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    
    # Anchors
    # Strides corresponding to N3, N4, N5 in our PANet
    strides = [8, 16, 32]
    base_sizes = [32, 64, 128]
    scales = [1, 2**(1/3), 2**(2/3)]
    aspect_ratios = [0.5, 1.0, 2.0]
    
    anchors = generate_anchors((512, 512), strides, base_sizes, scales, aspect_ratios)
    anchors = anchors.to(device)
    
    use_tal = getattr(args, 'use_tal', False)
    
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
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch_idx, (imgs, boxes, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            B = imgs.shape[0]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                pred_cls, pred_reg = model(imgs)
                
                # Prepare targets for the batch
                target_classes = []
                target_boxes = []
                
                for b in range(B):
                    gt_b = boxes[b].to(device)
                    gt_l = labels[b].to(device)
                    
                    if use_tal and gt_b.shape[0] > 0:
                        # Task-Aligned Assigner needs model predictions
                        t_cls, t_box = task_aligned_assign(
                            pred_cls[b].detach(),
                            pred_reg[b].detach(),
                            anchors, gt_b, gt_l
                        )
                    else:
                        # Legacy Max-IoU assigner (also used when no GTs)
                        t_cls, t_box = assign_targets(anchors, gt_b, gt_l)
                    
                    target_classes.append(t_cls)
                    target_boxes.append(t_box)
                    
                target_classes = torch.stack(target_classes, dim=0)
                target_boxes = torch.stack(target_boxes, dim=0)
                
                cls_loss, reg_loss, loss = criterion(pred_cls, pred_reg,
                                                     target_classes, target_boxes,
                                                     anchors)
            
            if loss > 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update EMA after each optimizer step
                ema.update(model)
                
                epoch_cls_loss += cls_loss.item()
                epoch_reg_loss += reg_loss.item()
                epoch_total_loss += loss.item()
                
            print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Step the learning rate scheduler
        scheduler.step()
        
        avg_loss = epoch_total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f} "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Extract state dict (handle DataParallel)
        model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        ema_state = ema.ema_model.module.state_dict() if isinstance(ema.ema_model, torch.nn.DataParallel) else ema.ema_model.state_dict()
        
        # Save checkpoints (always save EMA weights for best.pt)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'best_loss': best_loss,
        }
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
            
        torch.save(checkpoint, last_pt)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save EMA weights as best — these are the production weights
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': ema_state,
                'best_loss': best_loss,
            }
            torch.save(best_checkpoint, os.path.join(args.weights_dir, 'best.pt'))
            print("Saved new best model (EMA weights).")

if __name__ == '__main__':
    args = parse_args()
    train(args)
