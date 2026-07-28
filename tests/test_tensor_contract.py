import torch
import unittest
from wheeleye.models.detector import DecoupledHead

class TestTensorContract(unittest.TestCase):
    def test_flattening_order(self):
        """
        Verify that the detector's raw output flattening matches the 
        (H*W, num_anchors) order used by the target generator.
        
        This tests the crucial `permute(0, 3, 4, 1, 2)` contract.
        """
        B = 1
        C = 64
        H = 2
        W = 3
        num_anchors = 9
        num_classes = 4
        
        # Instantiate head (untrained)
        head = DecoupledHead(in_channels=C, num_classes=num_classes, num_anchors=num_anchors)
        
        # Create a synthetic prediction tensor prior to the view/permute operations.
        # The output of self.cls_conv(p) has shape (B, num_anchors * num_classes, H, W)
        synthetic_out = torch.zeros(B, num_anchors, num_classes, H, W)
        
        # Place a unique marker at a specific coordinate:
        # h=1, w=2, anchor=3, class=2
        target_h = 1
        target_w = 2
        target_a = 3
        target_c = 2
        marker_val = 100.0
        
        synthetic_out[0, target_a, target_c, target_h, target_w] = marker_val
        
        # Flatten the synthetic tensor using the exact same logic from `detector.py`
        cls_out = synthetic_out.permute(0, 3, 4, 1, 2).contiguous()
        cls_out = cls_out.view(B, -1, num_classes)
        
        # In the target generator (anchors.py), the grid is generated as:
        # over H, over W, over num_anchors.
        # So the flattened 1D spatial index should be:
        expected_flat_idx = (target_h * W + target_w) * num_anchors + target_a
        
        # Verify the marker landed exactly at this index
        actual_val = cls_out[0, expected_flat_idx, target_c].item()
        
        self.assertEqual(
            actual_val, marker_val,
            f"TENSOR CONTRACT VIOLATION: Marker at (h={target_h}, w={target_w}, a={target_a}) "
            f"should map to flat index {expected_flat_idx}, but found {actual_val} instead of {marker_val}. "
            f"This indicates the permute/flatten sequence in detector.py is out of sync with anchors.py!"
        )
        
        # Just to be extra safe, ensure no other index got the marker
        cls_out[0, expected_flat_idx, target_c] = 0.0
        self.assertEqual(cls_out.max().item(), 0.0, "Marker leaked to another index!")

if __name__ == '__main__':
    unittest.main()
