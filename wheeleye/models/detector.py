import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class Backbone(nn.Module):
    def __init__(self, freeze_early_layers=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        mobilenet = mobilenet_v3_small(weights=weights)
        
        # In MobileNetV3-Small, these indices roughly correspond to downsampling stages
        # We extract features at stride 8 (C3), 16 (C4), and 32 (C5)
        return_nodes = {
            'features.3': 'C3',   # Stride 8, 24 channels
            'features.8': 'C4',   # Stride 16, 48 channels
            'features.12': 'C5'   # Stride 32, 576 channels
        }
        
        self.body = create_feature_extractor(mobilenet, return_nodes=return_nodes)
        
        if freeze_early_layers:
            # Freeze layers up to features.3
            for name, param in self.body.named_parameters():
                if int(name.split('.')[1]) <= 3:
                    param.requires_grad = False
                    
        self.out_channels = {'C3': 24, 'C4': 48, 'C5': 576}

    def forward(self, x):
        return self.body(x)


class ConvBnSiLU(nn.Module):
    """Conv2d + BatchNorm2d + SiLU activation block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PANet(nn.Module):
    """
    Path Aggregation Network (PANet) — superior multi-scale feature fusion.

    Combines a top-down FPN pathway with a bottom-up path-aggregation pathway
    so that low-level spatial detail and high-level semantics can flow in both
    directions.  This dramatically improves small-object detection (fasteners,
    valve stems).

    Architecture:
        Top-down (FPN):  C5 → P5;  P5↑ + C4 → P4;  P4↑ + C3 → P3
        Bottom-up (PAN): P3 → N3;  N3↓ + P4 → N4;  N4↓ + P5 → N5
    """
    def __init__(self, in_channels_dict, out_channels=64):
        super().__init__()
        # ---- Top-down pathway (same as FPN) ----
        self.lat_c5 = nn.Conv2d(in_channels_dict['C5'], out_channels, 1)
        self.lat_c4 = nn.Conv2d(in_channels_dict['C4'], out_channels, 1)
        self.lat_c3 = nn.Conv2d(in_channels_dict['C3'], out_channels, 1)

        self.td_conv_p4 = ConvBnSiLU(out_channels, out_channels)
        self.td_conv_p3 = ConvBnSiLU(out_channels, out_channels)

        # ---- Bottom-up pathway (PANet addition) ----
        self.bu_down_n3 = ConvBnSiLU(out_channels, out_channels, stride=2, padding=1)
        self.bu_conv_n4 = ConvBnSiLU(out_channels, out_channels)

        self.bu_down_n4 = ConvBnSiLU(out_channels, out_channels, stride=2, padding=1)
        self.bu_conv_n5 = ConvBnSiLU(out_channels, out_channels)

    def forward(self, features):
        c3, c4, c5 = features['C3'], features['C4'], features['C5']

        # --- Top-down ---
        p5 = self.lat_c5(c5)

        p4 = self.lat_c4(c4)
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4 = p4 + p5_up
        p4 = self.td_conv_p4(p4)

        p3 = self.lat_c3(c3)
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p3 = p3 + p4_up
        p3 = self.td_conv_p3(p3)

        # --- Bottom-up ---
        n3 = p3  # N3 is just P3

        n3_down = self.bu_down_n3(n3)
        # Ensure spatial sizes match before addition
        if n3_down.shape[-2:] != p4.shape[-2:]:
            n3_down = F.interpolate(n3_down, size=p4.shape[-2:], mode='nearest')
        n4 = self.bu_conv_n4(n3_down + p4)

        n4_down = self.bu_down_n4(n4)
        if n4_down.shape[-2:] != p5.shape[-2:]:
            n4_down = F.interpolate(n4_down, size=p5.shape[-2:], mode='nearest')
        n5 = self.bu_conv_n5(n4_down + p5)

        return [n3, n4, n5]


class DecoupledHead(nn.Module):
    """
    Decoupled detection head for a single FPN level.

    Each FPN level gets its own independent head instance with separate
    classification and regression branches.  This ensures strict mathematical
    decoupling — no weight sharing across scales.

    Each branch: 2× (Conv3×3 + BN + SiLU) → Conv1×1 for output.
    """
    def __init__(self, in_channels=64, num_classes=4, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            ConvBnSiLU(in_channels, in_channels),
            ConvBnSiLU(in_channels, in_channels),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )
        
        # Bounding box regression branch (dx, dy, dw, dh)
        self.reg_conv = nn.Sequential(
            ConvBnSiLU(in_channels, in_channels),
            ConvBnSiLU(in_channels, in_channels),
            nn.Conv2d(in_channels, num_anchors * 4, 1)
        )
        
        # Initialization for Focal Loss
        # We initialize the bias of the cls_conv to -log((1-pi)/pi) with pi=0.01
        # This prevents the huge number of background anchors from dominating the loss in early iterations.
        pi = 0.01
        bias_val = -torch.math.log((1 - pi) / pi)
        nn.init.constant_(self.cls_conv[-1].bias, bias_val)

    def forward(self, p):
        """Process a single FPN level feature map."""
        B, C, H, W = p.shape
        
        # Classification output: (B, num_anchors * num_classes, H, W)
        cls_out = self.cls_conv(p)
        cls_out = cls_out.view(B, self.num_anchors, self.num_classes, H, W)
        cls_out = cls_out.permute(0, 3, 4, 1, 2).contiguous()
        cls_out = cls_out.view(B, -1, self.num_classes)
        
        # Regression output: (B, num_anchors * 4, H, W)
        reg_out = self.reg_conv(p)
        reg_out = reg_out.view(B, self.num_anchors, 4, H, W)
        reg_out = reg_out.permute(0, 3, 4, 1, 2).contiguous()
        reg_out = reg_out.view(B, -1, 4)
        
        return cls_out, reg_out


class WheelEyeDetector(nn.Module):
    def __init__(self, num_classes=4, num_anchors=9):
        super().__init__()
        self.backbone = Backbone(freeze_early_layers=True)
        self.neck = PANet(self.backbone.out_channels, out_channels=64)

        # Per-level decoupled heads — each FPN level gets its own independent
        # set of weights.  This is the "strict mathematical decoupling" from
        # the Phase 3 plan.
        self.heads = nn.ModuleList([
            DecoupledHead(in_channels=64, num_classes=num_classes, num_anchors=num_anchors)
            for _ in range(3)  # N3, N4, N5
        ])

    def forward(self, x):
        # x: (B, 3, 512, 512)
        features = self.backbone(x)
        pan_features = self.neck(features)

        cls_scores = []
        bbox_preds = []
        for head, feat in zip(self.heads, pan_features):
            cls, reg = head(feat)
            cls_scores.append(cls)
            bbox_preds.append(reg)

        return torch.cat(cls_scores, dim=1), torch.cat(bbox_preds, dim=1)
