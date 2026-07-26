import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask):
        """
        inputs: (N, C) logits
        targets: (N,) class indices
        mask: (N,) boolean mask indicating valid anchors (pos + neg)
        """
        # Filter inputs and targets based on mask
        inputs = inputs[mask]
        targets = targets[mask]
        
        # Targets are -2 for background, 0..C-1 for foreground
        # We need to convert targets to one-hot, and handle background.
        # Actually, standard Focal Loss in object detection uses sigmoid and one-hot targets.
        num_classes = inputs.shape[1]
        
        # one-hot targets: (N_valid, C)
        # Background (-2) gets all zeros. Foreground gets 1 at target class.
        pos_mask = targets >= 0
        
        t = torch.zeros_like(inputs)
        t[pos_mask, targets[pos_mask]] = 1.0
        
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, t, reduction='none')
        p_t = p * t + (1 - p) * (1 - t)
        
        alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss

def giou_loss(preds_cxcywh, targets_cxcywh):
    """
    Generalized Intersection over Union Loss
    Both inputs in (cx, cy, w, h) format.
    """
    # Convert to (x1, y1, x2, y2)
    b1_x1, b1_x2 = preds_cxcywh[:, 0] - preds_cxcywh[:, 2] / 2, preds_cxcywh[:, 0] + preds_cxcywh[:, 2] / 2
    b1_y1, b1_y2 = preds_cxcywh[:, 1] - preds_cxcywh[:, 3] / 2, preds_cxcywh[:, 1] + preds_cxcywh[:, 3] / 2
    
    b2_x1, b2_x2 = targets_cxcywh[:, 0] - targets_cxcywh[:, 2] / 2, targets_cxcywh[:, 0] + targets_cxcywh[:, 2] / 2
    b2_y1, b2_y2 = targets_cxcywh[:, 1] - targets_cxcywh[:, 3] / 2, targets_cxcywh[:, 1] + targets_cxcywh[:, 3] / 2
    
    # Intersection
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    
    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-16
    
    iou = inter_area / union_area
    
    # Convex Hull
    convex_x1 = torch.min(b1_x1, b2_x1)
    convex_y1 = torch.min(b1_y1, b2_y1)
    convex_x2 = torch.max(b1_x2, b2_x2)
    convex_y2 = torch.max(b1_y2, b2_y2)
    
    convex_area = torch.clamp(convex_x2 - convex_x1, min=0) * torch.clamp(convex_y2 - convex_y1, min=0) + 1e-16
    
    giou = iou - (convex_area - union_area) / convex_area
    
    loss = 1.0 - giou
    return loss.sum()

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=4, lambda_reg=1.0):
        super().__init__()
        self.cls_loss = FocalLoss()
        self.lambda_reg = lambda_reg

    def forward(self, pred_cls, pred_reg, target_classes, target_boxes, anchors):
        """
        pred_cls: (B, num_anchors, num_classes)
        pred_reg: (B, num_anchors, 4) - encoded offsets
        target_classes: (B, num_anchors) -> -1 for ignore, -2 for background, 0..C-1 for object
        target_boxes: (B, num_anchors, 4) -> unencoded gt boxes (cx, cy, w, h)
        anchors: (num_anchors, 4)
        """
        B = pred_cls.shape[0]
        device = pred_cls.device
        
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        num_pos = 0.0
        
        for b in range(B):
            valid_mask = target_classes[b] != -1
            pos_mask = target_classes[b] >= 0
            
            if valid_mask.sum() == 0:
                continue
                
            # Classification loss (on both positive and negative anchors)
            c_loss = self.cls_loss(pred_cls[b], target_classes[b], valid_mask)
            total_cls_loss += c_loss
            
            num_pos += pos_mask.sum().item()
            
            if pos_mask.sum() > 0:
                # Regression loss (only on positive anchors)
                from wheeleye.utils.anchors import decode_boxes
                decoded_boxes = decode_boxes(pred_reg[b][pos_mask], anchors[pos_mask])
                r_loss = giou_loss(decoded_boxes, target_boxes[b][pos_mask])
                
                # Also add smooth l1 for stability (optional, but requested in plan)
                # Actually GIoU is usually sufficient and stable. Let's combine smooth L1 on encoded offsets.
                from wheeleye.utils.anchors import encode_boxes
                encoded_gt = encode_boxes(target_boxes[b][pos_mask], anchors[pos_mask])
                l1_loss = F.smooth_l1_loss(pred_reg[b][pos_mask], encoded_gt, reduction='sum')
                
                total_reg_loss += r_loss + l1_loss
                
        # Normalize by number of positive anchors to ensure stability across batches
        num_pos = max(1.0, num_pos)
        
        cls_loss = total_cls_loss / num_pos
        reg_loss = (total_reg_loss / num_pos) * self.lambda_reg
        
        return cls_loss, reg_loss, cls_loss + reg_loss
