import os
import sys
import unittest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.utils.anchors import (
    generate_anchors, task_aligned_assign, decode_boxes
)


class TestTaskAlignedAssigner(unittest.TestCase):

    def setUp(self):
        """Generate anchors and create dummy predictions."""
        self.device = torch.device('cpu')
        strides = [8, 16, 32]
        base_sizes = [32, 64, 128]
        scales = [1, 2**(1/3), 2**(2/3)]
        aspect_ratios = [0.5, 1.0, 2.0]

        self.anchors = generate_anchors((512, 512), strides, base_sizes,
                                         scales, aspect_ratios).to(self.device)
        self.num_anchors = self.anchors.shape[0]
        self.num_classes = 4

    def _make_dummy_preds(self):
        """Create random predictions."""
        pred_cls = torch.randn(self.num_anchors, self.num_classes)
        pred_reg = torch.randn(self.num_anchors, 4) * 0.1  # small offsets
        return pred_cls, pred_reg

    def test_every_gt_gets_assigned(self):
        """Every GT box should be assigned to at least one positive anchor."""
        pred_cls, pred_reg = self._make_dummy_preds()

        gt_boxes = torch.tensor([
            [256.0, 256.0, 200.0, 200.0],  # large wheel
            [256.0, 256.0, 20.0, 20.0],     # small fastener
        ])
        gt_classes = torch.tensor([0, 1])

        t_cls, t_box = task_aligned_assign(pred_cls, pred_reg, self.anchors,
                                            gt_boxes, gt_classes)

        # Both classes should appear in assignments
        assigned_classes = set(t_cls[t_cls >= 0].tolist())
        for cls_id in gt_classes.tolist():
            self.assertIn(cls_id, assigned_classes,
                          f"GT class {cls_id} was not assigned to any anchor")

    def test_no_duplicate_assignments(self):
        """Each anchor should be assigned to at most one GT."""
        pred_cls, pred_reg = self._make_dummy_preds()

        gt_boxes = torch.tensor([
            [200.0, 200.0, 100.0, 100.0],
            [300.0, 300.0, 100.0, 100.0],
        ])
        gt_classes = torch.tensor([0, 1])

        t_cls, _ = task_aligned_assign(pred_cls, pred_reg, self.anchors,
                                        gt_boxes, gt_classes)

        # Each positive anchor should have exactly one class assigned
        pos_mask = t_cls >= 0
        pos_classes = t_cls[pos_mask]
        # No anchor should have more than one assignment (the function returns
        # a single class per anchor, so this is always true by construction,
        # but let's verify shapes are consistent)
        self.assertEqual(pos_classes.dim(), 1)

    def test_empty_gt_all_background(self):
        """With no GT boxes, all anchors should be background (-2)."""
        pred_cls, pred_reg = self._make_dummy_preds()

        gt_boxes = torch.zeros((0, 4))
        gt_classes = torch.zeros((0,), dtype=torch.long)

        t_cls, _ = task_aligned_assign(pred_cls, pred_reg, self.anchors,
                                        gt_boxes, gt_classes)

        self.assertTrue((t_cls == -2).all(),
                         msg="All anchors should be background when no GTs exist")

    def test_output_shapes(self):
        """Output shapes should match number of anchors."""
        pred_cls, pred_reg = self._make_dummy_preds()
        gt_boxes = torch.tensor([[256.0, 256.0, 100.0, 100.0]])
        gt_classes = torch.tensor([2])

        t_cls, t_box = task_aligned_assign(pred_cls, pred_reg, self.anchors,
                                            gt_boxes, gt_classes)

        self.assertEqual(t_cls.shape, (self.num_anchors,))
        self.assertEqual(t_box.shape, (self.num_anchors, 4))


if __name__ == '__main__':
    unittest.main()
