import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


def _bbox_cxcywh_to_xyxy(boxes):
    """Convert (cx, cy, w, h) boxes to (x1, y1, x2, y2) format."""
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return x1, y1, x2, y2


def ciou_loss(preds_cxcywh, targets_cxcywh):
    """
    Complete Intersection over Union (CIoU) Loss.
    
    Penalises three terms simultaneously:
      1. IoU overlap
      2. Centre-point distance (normalised by enclosing-box diagonal)
      3. Aspect-ratio consistency
    
    Both inputs in (cx, cy, w, h) format.  Returns scalar loss (sum-reduced).
    """
    eps = 1e-7

    # Convert to xyxy
    b1_x1, b1_y1, b1_x2, b1_y2 = _bbox_cxcywh_to_xyxy(preds_cxcywh)
    b2_x1, b2_y1, b2_x2, b2_y2 = _bbox_cxcywh_to_xyxy(targets_cxcywh)

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + eps

    iou = inter_area / union_area

    # ---- Centre distance penalty ----
    # Squared distance between centres
    centre_dist_sq = (preds_cxcywh[..., 0] - targets_cxcywh[..., 0]) ** 2 + \
                     (preds_cxcywh[..., 1] - targets_cxcywh[..., 1]) ** 2

    # Diagonal² of smallest enclosing box
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    rho = centre_dist_sq / enclose_diag_sq

    # ---- Aspect-ratio penalty ----
    w_pred = preds_cxcywh[..., 2]
    h_pred = preds_cxcywh[..., 3]
    w_gt = targets_cxcywh[..., 2]
    h_gt = targets_cxcywh[..., 3]

    v = (4 / (math.pi ** 2)) * (
        torch.atan(w_gt / (h_gt + eps)) - torch.atan(w_pred / (h_pred + eps))
    ) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # CIoU
    ciou = iou - rho - alpha * v
    loss = 1.0 - ciou

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
                # Decode predicted boxes to absolute coordinates
                from wheeleye.utils.anchors import decode_boxes
                decoded_boxes = decode_boxes(pred_reg[b][pos_mask], anchors[pos_mask])
                
                # CIoU loss — operates directly on decoded boxes, no need for
                # encoded-offset Smooth L1 (CIoU already penalises centre, size,
                # and aspect ratio in a unified way).
                r_loss = ciou_loss(decoded_boxes, target_boxes[b][pos_mask])
                total_reg_loss += r_loss
                
        # Normalize by number of positive anchors to ensure stability across batches
        num_pos = max(1.0, num_pos)
        
        cls_loss = total_cls_loss / num_pos
        reg_loss = (total_reg_loss / num_pos) * self.lambda_reg
        
        return cls_loss, reg_loss, cls_loss + reg_loss
