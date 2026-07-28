import os
import sys
import unittest
import torch
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.utils.loss import ciou_loss


class TestCIoULoss(unittest.TestCase):

    def test_identical_boxes_zero_loss(self):
        """CIoU of identical boxes should be 1.0 → loss = 0.0."""
        boxes = torch.tensor([[100.0, 100.0, 50.0, 50.0]])
        loss = ciou_loss(boxes, boxes)
        self.assertAlmostEqual(loss.item(), 0.0, places=5,
                               msg="Identical boxes should produce zero CIoU loss")

    def test_non_overlapping_boxes_high_loss(self):
        """Non-overlapping boxes should produce loss > 1.0."""
        pred = torch.tensor([[10.0, 10.0, 5.0, 5.0]])
        gt = torch.tensor([[200.0, 200.0, 5.0, 5.0]])
        loss = ciou_loss(pred, gt)
        self.assertGreater(loss.item(), 1.0,
                           msg="Non-overlapping boxes should produce high CIoU loss")

    def test_partial_overlap_moderate_loss(self):
        """Partially overlapping boxes should produce 0 < loss < 2."""
        pred = torch.tensor([[100.0, 100.0, 50.0, 50.0]])
        gt = torch.tensor([[120.0, 120.0, 50.0, 50.0]])
        loss = ciou_loss(pred, gt)
        self.assertGreater(loss.item(), 0.0)
        self.assertLess(loss.item(), 2.0)

    def test_gradient_flows(self):
        """Verify gradients flow through CIoU without NaN or Inf."""
        pred = torch.tensor([[100.0, 100.0, 50.0, 50.0]], requires_grad=True)
        gt = torch.tensor([[110.0, 110.0, 60.0, 40.0]])
        loss = ciou_loss(pred, gt)
        loss.backward()
        self.assertFalse(torch.isnan(pred.grad).any(),
                         msg="Gradient should not contain NaN")
        self.assertFalse(torch.isinf(pred.grad).any(),
                         msg="Gradient should not contain Inf")

    def test_batch_reduction(self):
        """CIoU loss should be sum-reduced over batch dimension."""
        pred = torch.tensor([[100.0, 100.0, 50.0, 50.0],
                              [200.0, 200.0, 30.0, 30.0]])
        gt = torch.tensor([[100.0, 100.0, 50.0, 50.0],
                            [210.0, 210.0, 30.0, 30.0]])
        loss = ciou_loss(pred, gt)
        # First box is identical (loss=0), second has slight offset (loss>0)
        self.assertGreater(loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
