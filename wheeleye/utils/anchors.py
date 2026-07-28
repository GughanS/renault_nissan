import torch
import math

def generate_anchors(image_size, strides, base_sizes, scales, aspect_ratios):
    """
    Generate anchor boxes across all FPN levels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_anchors = []
    
    for stride, base_size in zip(strides, base_sizes):
        feat_h = image_size[0] // stride
        feat_w = image_size[1] // stride
        
        # Grid of center coordinates
        shifts_x = torch.arange(0, feat_w, device=device) * stride + stride / 2
        shifts_y = torch.arange(0, feat_h, device=device) * stride + stride / 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        
        # Anchor dimensions for this feature level
        anchors_wh = []
        for scale in scales:
            for ar in aspect_ratios:
                w = base_size * scale * torch.sqrt(torch.tensor(ar, device=device))
                h = base_size * scale / torch.sqrt(torch.tensor(ar, device=device))
                anchors_wh.append([w, h])
                
        anchors_wh = torch.tensor(anchors_wh, dtype=torch.float32, device=device)
        num_anchors = len(anchors_wh)
        
        # Broadcast and combine
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1) # (H*W, 4)
        shifts = shifts.unsqueeze(1).repeat(1, num_anchors, 1) # (H*W, num_anchors, 4)
        
        anchors_wh = anchors_wh.unsqueeze(0).repeat(feat_h * feat_w, 1, 1) # (H*W, num_anchors, 2)
        
        # (cx, cy, w, h)
        anchors_cxcywh = torch.zeros_like(shifts)
        anchors_cxcywh[..., 0:2] = shifts[..., 0:2]
        anchors_cxcywh[..., 2:4] = anchors_wh
        
        all_anchors.append(anchors_cxcywh.view(-1, 4))
        
    return torch.cat(all_anchors, dim=0) # (Total_Anchors, 4)

def bbox_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    Boxes should be in (cx, cy, w, h) format.
    Returns: (N, M) IoU matrix where N = len(box1), M = len(box2).
    """
    # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # Intersection area
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = b1_area.unsqueeze(1) + b2_area - inter_area + 1e-16
    
    return inter_area / union_area


def _is_anchor_centre_in_gt(anchors, gt_boxes):
    """
    Check whether each anchor's centre falls inside each GT box.
    anchors: (A, 4)  gt_boxes: (G, 4)  both in (cx, cy, w, h).
    Returns: (A, G) boolean mask.
    """
    a_cx = anchors[:, 0].unsqueeze(1)  # (A, 1)
    a_cy = anchors[:, 1].unsqueeze(1)

    gt_x1 = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2).unsqueeze(0)  # (1, G)
    gt_y1 = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2).unsqueeze(0)
    gt_x2 = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2).unsqueeze(0)
    gt_y2 = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2).unsqueeze(0)

    in_x = (a_cx >= gt_x1) & (a_cx <= gt_x2)
    in_y = (a_cy >= gt_y1) & (a_cy <= gt_y2)

    return in_x & in_y  # (A, G)


def task_aligned_assign(pred_cls, pred_reg, anchors, gt_boxes, gt_classes,
                        topk=13, alpha=1.0, beta=6.0):
    """
    Task-Aligned Assigner (TAL) for dynamic anchor-GT matching.

    Unlike Max-IoU assignment, TAL uses both predicted classification quality
    and predicted box quality (IoU) to decide which anchors are best aligned
    with each GT — making assignment prediction-aware and eliminating static
    scaling bugs.

    Args:
        pred_cls: (num_anchors, num_classes) — sigmoid classification logits
        pred_reg: (num_anchors, 4) — predicted encoded offsets
        anchors:  (num_anchors, 4) — anchor boxes (cx, cy, w, h)
        gt_boxes: (num_gt, 4) — ground-truth boxes (cx, cy, w, h)
        gt_classes: (num_gt,) — class indices for each GT
        topk: number of top-aligned anchors per GT to consider
        alpha: exponent for classification alignment score
        beta:  exponent for IoU alignment score

    Returns:
        target_classes: (num_anchors,)  -1 = ignore, -2 = background, 0..C-1 = assigned
        target_boxes:   (num_anchors, 4)
    """
    num_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]
    device = anchors.device

    if num_gt == 0:
        return (torch.zeros(num_anchors, dtype=torch.long, device=device) - 2,
                torch.zeros_like(anchors))

    # --- Step 1: Compute alignment metric t = cls_score^α × IoU^β ---
    # Get predicted classification scores for each GT's class
    cls_scores = torch.sigmoid(pred_cls)  # (A, C)
    gt_cls_scores = cls_scores[:, gt_classes]  # (A, G) — score for each GT's class

    # Decode predicted boxes and compute IoU with each GT
    decoded_preds = decode_boxes(pred_reg, anchors)  # (A, 4)
    ious = bbox_iou(decoded_preds, gt_boxes)  # (A, G)

    # Alignment metric
    alignment = gt_cls_scores.pow(alpha) * ious.pow(beta)  # (A, G)

    # --- Step 2: Spatial prior — only anchors whose centre is inside GT ---
    centre_mask = _is_anchor_centre_in_gt(anchors, gt_boxes)  # (A, G)
    alignment = alignment * centre_mask.float()

    # --- Step 3: Select top-k anchors per GT ---
    effective_topk = min(topk, num_anchors)
    topk_metrics, topk_idxs = alignment.topk(effective_topk, dim=0, largest=True)  # (K, G)

    # Build a candidate mask: (A, G) — True where anchor is in a GT's top-k
    candidate_mask = torch.zeros_like(alignment, dtype=torch.bool)
    for g in range(num_gt):
        candidate_mask[topk_idxs[:, g], g] = True

    # Also enforce centre prior on candidates
    candidate_mask = candidate_mask & centre_mask

    # --- Step 4: Resolve conflicts — if multiple GTs claim the same anchor,
    #             assign to the GT with the highest IoU ---
    # Count how many GTs claim each anchor
    num_claims = candidate_mask.sum(dim=1)  # (A,)
    conflict_mask = num_claims > 1

    if conflict_mask.any():
        # For conflicting anchors, pick the GT with the highest IoU
        conflict_ious = ious[conflict_mask] * candidate_mask[conflict_mask].float()  # (C, G)
        best_gt = conflict_ious.argmax(dim=1)  # (C,)
        # Clear all claims, keep only the best
        new_candidates = torch.zeros(conflict_mask.sum(), num_gt, dtype=torch.bool, device=device)
        new_candidates[torch.arange(len(best_gt), device=device), best_gt] = True
        candidate_mask[conflict_mask] = new_candidates

    # --- Step 5: Build final assignments ---
    # Each anchor is assigned to at most one GT
    is_positive = candidate_mask.any(dim=1)  # (A,)
    assigned_gt = candidate_mask.float().argmax(dim=1)  # (A,) — GT index for each anchor

    target_classes = torch.zeros(num_anchors, dtype=torch.long, device=device) - 2  # default: bg
    target_boxes = torch.zeros_like(anchors)

    target_classes[is_positive] = gt_classes[assigned_gt[is_positive]]
    target_boxes[is_positive] = gt_boxes[assigned_gt[is_positive]]

    # Anchors between pos and neg that aren't in any top-k: leave as -2 (background)
    # Anchors that are NOT positive and NOT clearly negative can be set to ignore (-1)
    # but for TAL, standard practice is: positive = assigned, everything else = background.

    return target_classes, target_boxes


def assign_targets(anchors, gt_boxes, gt_classes, pos_iou_thr=0.5, neg_iou_thr=0.4):
    """
    Legacy assigner: Assign ground truth boxes and classes to anchors using
    Max IoU bipartite matching.

    Kept for backward compatibility. New training code should use
    ``task_aligned_assign`` instead.
    """
    num_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]
    device = anchors.device
    
    if num_gt == 0:
        return torch.zeros(num_anchors, dtype=torch.long, device=device) - 1, torch.zeros_like(anchors)
    
    ious = bbox_iou(anchors, gt_boxes) # (num_anchors, num_gt)
    
    # Max IoU for each anchor
    max_ious, argmax_ious = ious.max(dim=1)
    
    # Max IoU for each GT box (ensure each GT box gets at least one anchor)
    gt_max_ious, gt_argmax_ious = ious.max(dim=0)
    
    # Target labels and boxes initialization
    target_classes = torch.zeros(num_anchors, dtype=torch.long, device=device) - 1 # -1 is background/ignore
    target_boxes = torch.zeros_like(anchors)
    
    pos_mask = max_ious >= pos_iou_thr
    neg_mask = max_ious < neg_iou_thr
    
    # Ensure every GT box is assigned to its best matching anchor
    for i in range(num_gt):
        if gt_max_ious[i] > 0:
            pos_mask[gt_argmax_ious[i]] = True
            neg_mask[gt_argmax_ious[i]] = False
            argmax_ious[gt_argmax_ious[i]] = i
            
    # Assign targets
    target_classes[pos_mask] = gt_classes[argmax_ious[pos_mask]]
    target_boxes[pos_mask] = gt_boxes[argmax_ious[pos_mask]]
    
    # Background
    target_classes[neg_mask] = -2 # -2 means negative anchor (background)
    
    return target_classes, target_boxes

def encode_boxes(gt_boxes, anchors):
    """
    Encode GT boxes relative to anchors for bounding box regression.
    (cx, cy, w, h)
    """
    encoded_cxcy = (gt_boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
    encoded_wh = torch.log(gt_boxes[:, 2:] / anchors[:, 2:])
    return torch.cat([encoded_cxcy, encoded_wh], dim=1)

def decode_boxes(preds, anchors):
    """
    Decode predicted offsets to actual bounding boxes.
    """
    pred_cxcy = preds[:, :2] * anchors[:, 2:] + anchors[:, :2]
    pred_wh = torch.exp(preds[:, 2:]) * anchors[:, 2:]
    return torch.cat([pred_cxcy, pred_wh], dim=1)
