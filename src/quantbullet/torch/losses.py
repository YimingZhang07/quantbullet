import torch
import torch.nn as nn
import numpy as np

from quantbullet.torch.hinge import Hinge


class EpsilonInsensitiveLoss(nn.Module):
    """
    Epsilon-insensitive loss (SVR-style).

    Penalty rule:
      - |pred - target| <= epsilon  -->  loss = 0        (within tolerance, no penalty)
      - |pred - target| >  epsilon  -->  loss = |error| - epsilon  (linear penalty on excess)

    Compared to MAE: introduces a "dead zone" of width 2*epsilon centred on the target.
    Only prediction errors exceeding epsilon contribute to the gradient,
    making the model tolerant to small noise (e.g. bid-ask noise in spread data).
    """

    def __init__(self, epsilon=0.10):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        return torch.clamp(error - self.epsilon, min=0.0).mean()