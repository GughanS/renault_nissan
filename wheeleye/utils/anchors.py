import torch

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

def assign_targets(anchors, gt_boxes, gt_classes, pos_iou_thr=0.5, neg_iou_thr=0.4):
    """
    Assign ground truth boxes and classes to anchors using Max IoU bipartite matching.
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
    
    # Background anchors (IoU < neg_iou_thr) are labeled as 0 
    # Wait, in our loss, class 0 might be a valid class. Let's say valid classes are 0..C-1
    # We use num_classes as background in the Focal Loss, or we use a separate mask.
    # Let's use -1 for ignore, num_classes (or 0 if one-hot) for background.
    # For now, let's return pos_mask and neg_mask.
    
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
