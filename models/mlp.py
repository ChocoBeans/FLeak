"""Small MLP: one hidden layer. Outputs a raw logit (apply a sigmoid for probabilities)."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)
