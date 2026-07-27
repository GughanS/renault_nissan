import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from wheeleye.models.classifier import WheelEyeClassifier

def test_classifier_smoke():
    # 1. Initialize model
    model = WheelEyeClassifier(num_material=2, num_tier=3, num_size=3)
    
    # 2. Create dummy input batch (B=4, C=3, H=224, W=224)
    dummy_imgs = torch.randn(4, 3, 224, 224)
    
    # 3. Create dummy targets
    target_mat = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    target_tier = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    target_size = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    
    # 4. Forward pass
    out_mat, out_tier, out_size = model(dummy_imgs)
    
    # Check shapes
    assert out_mat.shape == (4, 2), f"Expected (4, 2), got {out_mat.shape}"
    assert out_tier.shape == (4, 3), f"Expected (4, 3), got {out_tier.shape}"
    assert out_size.shape == (4, 3), f"Expected (4, 3), got {out_size.shape}"
    
    # 5. Loss calculation
    criterion = nn.CrossEntropyLoss()
    loss_mat = criterion(out_mat, target_mat)
    loss_tier = criterion(out_tier, target_tier)
    loss_size = criterion(out_size, target_size)
    
    total_loss = loss_mat + loss_tier + loss_size
    
    # 6. Backward pass
    total_loss.backward()
    
    # Check if gradients exist for a head (e.g. material_head)
    assert model.material_head.weight.grad is not None, "Backward pass failed, no gradients for material_head."
    
    print("Smoke test passed! WheelEyeClassifier forward and backward passes work correctly.")

if __name__ == '__main__':
    test_classifier_smoke()
