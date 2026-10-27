"""Logistic regression: one linear layer. Outputs a raw logit (apply a sigmoid for probabilities)."""
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
