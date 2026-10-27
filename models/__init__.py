"""Models shared by the centralized and federated runs (train with BCEWithLogitsLoss)."""
from models.logistic_regression import LogisticRegression
from models.mlp import MLP


def build_model(name, n_features, **kwargs):
    """`name` is "logreg" or "mlp"; the input size is `n_features`."""
    return {"logreg": LogisticRegression, "mlp": MLP}[name](n_features, **kwargs)
