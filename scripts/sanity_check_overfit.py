import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wheeleye.models.detector import WheelEyeDetector
from wheeleye.datasets.yolo_dataset import YOLODataset, collate_fn
from wheeleye.utils.loss import DetectionLoss
from wheeleye.utils.anchors import generate_anchors, task_aligned_assign, decode_boxes

def plot_boxes(img_tensor, gt_boxes, pred_boxes, filename):
    # Un-normalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img_disp = img_tensor.cpu() * std + mean
    img_disp = img_disp.clamp(0, 1).permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_disp)
    
    # Plot GT in green
    for box in gt_boxes:
        cx, cy, w, h = box.cpu().numpy()
        x = cx - w / 2
        y = cy - h / 2
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        
    # Plot Preds in red (only confident ones)
    for box in pred_boxes:
        cx, cy, w, h = box.cpu().numpy()
        x = cx - w / 2
        y = cy - h / 2
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
    plt.savefig(filename)
    plt.close()

def run_sanity_check():
    print("Running tiny-batch overfit sanity check...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = YOLODataset('data/images', 'data/labels', img_size=(512, 512), 
                          augment=False, mosaic=False)
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, 
                            collate_fn=collate_fn)
    
    # Get exactly one batch
    imgs, boxes, labels = next(iter(dataloader))
    imgs = imgs.to(device)
    
    model = WheelEyeDetector().to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = DetectionLoss(num_classes=4)
    
    strides = [8, 16, 32]
    base_sizes = [32, 64, 128]
    scales = [1, 2**(1/3), 2**(2/3)]
    aspect_ratios = [0.5, 1.0, 2.0]
    anchors = generate_anchors((512, 512), strides, base_sizes, scales, aspect_ratios).to(device)
    
    losses = []
    
    # Train on this single batch for 50 steps
    for step in range(50):
        optimizer.zero_grad()
        pred_cls, pred_reg = model(imgs)
        
        target_classes = []
        target_boxes = []
        
        for b in range(imgs.shape[0]):
            gt_b = boxes[b].to(device)
            gt_l = labels[b].to(device)
            
            if gt_b.shape[0] > 0:
                t_cls, t_box = task_aligned_assign(
                    pred_cls[b].detach(), pred_reg[b].detach(), anchors, gt_b, gt_l
                )
            else:
                t_cls = torch.zeros_like(pred_cls[b])
                t_box = torch.zeros_like(pred_reg[b])
                
            target_classes.append(t_cls)
            target_boxes.append(t_box)
            
        target_classes = torch.stack(target_classes, dim=0)
        target_boxes = torch.stack(target_boxes, dim=0)
        
        cls_loss, reg_loss, loss = criterion(pred_cls, pred_reg, target_classes, target_boxes, anchors)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/50, Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, Reg: {reg_loss.item():.4f})")
            
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    
    # Verification: Did loss drop significantly?
    assert losses[-1] < losses[0] * 0.1, "Loss did not drop by 90% -- model failed to overfit tiny batch!"
    print("Sanity check passed! Model successfully overfit the tiny batch.")
    
    # Visualization
    model.eval()
    with torch.no_grad():
        pred_cls, pred_reg = model(imgs)
        pred_cls = torch.sigmoid(pred_cls)
        
        # Visualize first image in batch
        b_idx = 0
        conf_mask = pred_cls[b_idx].max(dim=1)[0] > 0.5
        decoded_boxes = decode_boxes(pred_reg[b_idx], anchors)
        confident_boxes = decoded_boxes[conf_mask]
        
        os.makedirs('exports_test', exist_ok=True)
        plot_boxes(imgs[b_idx], boxes[b_idx], confident_boxes, 'exports_test/sanity_check_boxes.png')
        print("Saved visualization to exports_test/sanity_check_boxes.png")

if __name__ == '__main__':
    run_sanity_check()
