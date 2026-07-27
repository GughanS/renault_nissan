import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class WheelEyeClassifier(nn.Module):
    def __init__(self, num_material=2, num_tier=3, num_size=3):
        super().__init__()
        
        # Load pre-trained MobileNetV3-Small
        weights = MobileNet_V3_Small_Weights.DEFAULT
        mobilenet = mobilenet_v3_small(weights=weights)
        
        # Extract features (remove the original classifier)
        self.features = mobilenet.features
        
        # We also want the pooling layer and the first part of the original classifier (which is a linear + hardswish + dropout)
        # However, for a simple custom multi-head, we can just use our own Global Average Pooling and simple Linear heads.
        # MobileNetV3 small features output is shape (B, 576, H, W)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        in_features = 576
        
        # Multi-head branches
        self.material_head = nn.Linear(in_features, num_material)
        self.tier_head = nn.Linear(in_features, num_tier)
        self.size_head = nn.Linear(in_features, num_size)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        
        material_out = self.material_head(x)
        tier_out = self.tier_head(x)
        size_out = self.size_head(x)
        
        return material_out, tier_out, size_out
