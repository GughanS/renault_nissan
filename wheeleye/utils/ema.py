import math
import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model parameters.

    Maintains a shadow copy of the model weights and updates them after each
    optimiser step using:

        ema_weight = decay × ema_weight + (1 − decay) × model_weight

    The decay ramps up from ~0.9 to ``max_decay`` over training so that early,
    rapidly-changing weights are tracked more closely, while late-stage
    fine-tuning is heavily smoothed.

    EMA weights typically achieve higher peak accuracy and better stability
    than the raw training weights.  ``best.pt`` should always be saved from
    the EMA state dict.
    """

    def __init__(self, model: nn.Module, max_decay: float = 0.9999,
                 warmup_steps: int = 2000):
        # Deep-copy the model to create the shadow
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.max_decay = max_decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def _decay(self) -> float:
        """Ramp decay from low to max_decay so early updates track closely."""
        return min(self.max_decay,
                   (1 + self.step_count) / (self.warmup_steps + self.step_count))

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights from the live model after an optimiser step."""
        d = self._decay()
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1 - d)
        # Also update buffers (BN running_mean / running_var)
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)
        self.step_count += 1

    def state_dict(self):
        """Serialise EMA state for checkpointing."""
        return {
            'ema_model_state_dict': self.ema_model.state_dict(),
            'step_count': self.step_count,
        }

    def load_state_dict(self, state):
        """Restore EMA state from a checkpoint."""
        self.ema_model.load_state_dict(state['ema_model_state_dict'])
        self.step_count = state['step_count']
