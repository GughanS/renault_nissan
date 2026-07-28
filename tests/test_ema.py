import os
import sys
import unittest
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.utils.ema import ModelEMA


class SimpleModel(nn.Module):
    """Tiny model for EMA testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.bn = nn.BatchNorm1d(5)

    def forward(self, x):
        return self.bn(self.linear(x))


class TestModelEMA(unittest.TestCase):

    def setUp(self):
        self.model = SimpleModel()

    def test_ema_diverges_after_update(self):
        """After gradient updates + EMA updates, EMA weights should lag behind model."""
        # High decay with long warmup → EMA barely moves, model moves a lot
        ema = ModelEMA(self.model, max_decay=0.9999, warmup_steps=10000)
        ema.step_count = 100000  # Force decay to be max_decay

        optimizer = torch.optim.SGD(self.model.parameters(), lr=5.0)

        # Snapshot EMA weights before any update
        ema_w_before = ema.ema_model.linear.weight.data.clone()

        # Run updates: manually move the model weights far from initial
        for _ in range(10):
            for p in self.model.parameters():
                p.data.add_(10.0)
            ema.update(self.model)

        model_w = self.model.linear.weight.data
        ema_w = ema.ema_model.linear.weight.data

        # EMA should have moved from its initial position
        self.assertFalse(torch.allclose(ema_w, ema_w_before, atol=1e-6),
                         msg="EMA should update from initial weights")

        # But EMA should NOT have caught up to the model (it lags)
        diff = (model_w - ema_w).abs().max().item()
        self.assertGreater(diff, 0.01,
                           msg="EMA weights should lag behind rapidly moving model weights")

    def test_ema_converges_over_many_steps(self):
        """EMA should converge toward the model weights over many steps."""
        ema = ModelEMA(self.model, max_decay=0.99, warmup_steps=5)

        # Keep model weights fixed, update EMA many times
        for _ in range(200):
            ema.update(self.model)

        model_w = self.model.linear.weight.data
        ema_w = ema.ema_model.linear.weight.data
        self.assertTrue(torch.allclose(model_w, ema_w, atol=1e-3),
                        msg="EMA should converge toward model weights after many steps")

    def test_state_dict_roundtrip(self):
        """Saving and loading EMA state should produce identical weights."""
        ema = ModelEMA(self.model)
        # Do a few updates
        for _ in range(5):
            ema.update(self.model)

        state = ema.state_dict()

        # Create new EMA and load state
        ema2 = ModelEMA(self.model)
        ema2.load_state_dict(state)

        w1 = ema.ema_model.linear.weight.data
        w2 = ema2.ema_model.linear.weight.data
        self.assertTrue(torch.equal(w1, w2),
                        msg="EMA state dict roundtrip should be lossless")
        self.assertEqual(ema.step_count, ema2.step_count)

    def test_step_count_increments(self):
        """Step count should increment with each update."""
        ema = ModelEMA(self.model)
        self.assertEqual(ema.step_count, 0)
        ema.update(self.model)
        self.assertEqual(ema.step_count, 1)
        ema.update(self.model)
        self.assertEqual(ema.step_count, 2)


if __name__ == '__main__':
    unittest.main()
