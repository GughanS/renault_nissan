import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class WheelEyeClassifier(nn.Module):
    def __init__(self, num_material=2, num_tier=3, num_size=3):
        super().__init__()
        
        # Load pre-trained MobileNetV3-Large
        weights = MobileNet_V3_Large_Weights.DEFAULT
        mobilenet = mobilenet_v3_large(weights=weights)
        
        # Extract features (remove the original classifier)
        self.features = mobilenet.features
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        
        in_features = 960
        
        # Multi-head branches
        self.material_head = nn.Linear(in_features, num_material)
        self.tier_head = nn.Linear(in_features, num_tier)
        self.size_head = nn.Linear(in_features, num_size)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        
        material_out = self.material_head(x)
        tier_out = self.tier_head(x)
        size_out = self.size_head(x)
        
        return material_out, tier_out, size_out
