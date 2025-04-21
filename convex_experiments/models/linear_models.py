import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    """Simple linear regression model."""
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)

class LogisticRegressionModel(nn.Module):
    """Simple logistic regression model."""
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1) # Output is logits

    def forward(self, x):
        # Return logits, loss function (BCEWithLogitsLoss) will handle sigmoid
        return self.linear(x) 