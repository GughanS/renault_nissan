import os
import sys
import unittest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.models.detector import WheelEyeDetector


class TestPANet(unittest.TestCase):

    def setUp(self):
        """Create model on CPU."""
        self.model = WheelEyeDetector(num_classes=4, num_anchors=9)
        self.model.eval()

    def test_forward_pass_shape(self):
        """Smoke test: forward pass produces correct output shapes."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 512, 512)

        with torch.no_grad():
            cls_scores, bbox_preds = self.model(x)

        # Expected total anchors:
        # Stride 8:  (512/8)²  × 9 = 64² × 9 = 36864
        # Stride 16: (512/16)² × 9 = 32² × 9 = 9216
        # Stride 32: (512/32)² × 9 = 16² × 9 = 2304
        # Total = 48384
        expected_anchors = 36864 + 9216 + 2304

        self.assertEqual(cls_scores.shape, (batch_size, expected_anchors, 4),
                         f"cls_scores shape mismatch: {cls_scores.shape}")
        self.assertEqual(bbox_preds.shape, (batch_size, expected_anchors, 4),
                         f"bbox_preds shape mismatch: {bbox_preds.shape}")

    def test_per_level_heads_are_independent(self):
        """Each FPN level should have its own independent head weights."""
        # Compare first conv weight of each head — they should NOT share memory
        head_weights = [head.cls_conv[0].conv.weight for head in self.model.heads]
        for i in range(len(head_weights)):
            for j in range(i + 1, len(head_weights)):
                self.assertFalse(
                    head_weights[i].data_ptr() == head_weights[j].data_ptr(),
                    f"Heads {i} and {j} share weight memory — they should be independent"
                )

    def test_output_channels(self):
        """Verify PANet outputs 64 channels at each level."""
        x = torch.randn(1, 3, 512, 512)
        features = self.model.backbone(x)
        pan_features = self.model.neck(features)

        self.assertEqual(len(pan_features), 3)
        for i, feat in enumerate(pan_features):
            self.assertEqual(feat.shape[1], 64,
                             f"PANet level {i} has {feat.shape[1]} channels, expected 64")


if __name__ == '__main__':
    unittest.main()
