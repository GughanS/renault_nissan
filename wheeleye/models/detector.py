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


class FPN(nn.Module):
    def __init__(self, in_channels_dict, out_channels=64):
        super().__init__()
        # Lateral convolutions to reduce/standardize channel counts
        self.lat_c5 = nn.Conv2d(in_channels_dict['C5'], out_channels, 1)
        self.lat_c4 = nn.Conv2d(in_channels_dict['C4'], out_channels, 1)
        self.lat_c3 = nn.Conv2d(in_channels_dict['C3'], out_channels, 1)
        
        # Anti-aliasing convolutions after upsampling + addition
        self.conv_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features):
        c3, c4, c5 = features['C3'], features['C4'], features['C5']
        
        p5 = self.lat_c5(c5)
        
        p4 = self.lat_c4(c4)
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4 = p4 + p5_up
        p4 = self.conv_p4(p4)
        
        p3 = self.lat_c3(c3)
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p3 = p3 + p4_up
        p3 = self.conv_p3(p3)
        
        return [p3, p4, p5]


class DecoupledHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=4, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )
        
        # Bounding box regression branch (dx, dy, dw, dh)
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, 1)
        )
        
        # Initialization for Focal Loss
        # We initialize the bias of the cls_conv to -log((1-pi)/pi) with pi=0.01
        # This prevents the huge number of background anchors from dominating the loss in early iterations.
        pi = 0.01
        bias_val = -torch.math.log((1 - pi) / pi)
        nn.init.constant_(self.cls_conv[-1].bias, bias_val)

    def forward(self, features):
        cls_scores = []
        bbox_preds = []
        
        for p in features:
            B, C, H, W = p.shape
            
            # Classification output: (B, num_anchors * num_classes, H, W)
            cls_out = self.cls_conv(p)
            cls_out = cls_out.view(B, self.num_anchors, self.num_classes, H, W)
            cls_out = cls_out.permute(0, 1, 3, 4, 2).contiguous()
            cls_scores.append(cls_out.view(B, -1, self.num_classes))
            
            # Regression output: (B, num_anchors * 4, H, W)
            reg_out = self.reg_conv(p)
            reg_out = reg_out.view(B, self.num_anchors, 4, H, W)
            reg_out = reg_out.permute(0, 1, 3, 4, 2).contiguous()
            bbox_preds.append(reg_out.view(B, -1, 4))
            
        return torch.cat(cls_scores, dim=1), torch.cat(bbox_preds, dim=1)


class WheelEyeDetector(nn.Module):
    def __init__(self, num_classes=4, num_anchors=9):
        super().__init__()
        self.backbone = Backbone(freeze_early_layers=True)
        self.neck = FPN(self.backbone.out_channels, out_channels=64)
        self.head = DecoupledHead(in_channels=64, num_classes=num_classes, num_anchors=num_anchors)

    def forward(self, x):
        # x: (B, 3, 512, 512)
        features = self.backbone(x)
        fpn_features = self.neck(features)
        cls_scores, bbox_preds = self.head(fpn_features)
        
        return cls_scores, bbox_preds
