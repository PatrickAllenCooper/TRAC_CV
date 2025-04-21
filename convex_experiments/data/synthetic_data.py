import torch
import numpy as np

def generate_linear_data(num_samples, num_features, noise_std=0.1, seed=None):
    """Generates synthetic data for linear regression.

    Args:
        num_samples (int): Number of data points.
        num_features (int): Number of features.
        noise_std (float): Standard deviation of Gaussian noise added to y.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (X, y, w_true)
            X (torch.Tensor): Feature matrix (num_samples, num_features).
            y (torch.Tensor): Target vector (num_samples, 1).
            w_true (torch.Tensor): True weight vector (num_features, 1).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate true weights (add bias term implicitly later if needed or explicitly here)
    w_true = torch.randn(num_features, 1)

    # Generate features
    X = torch.randn(num_samples, num_features)

    # Generate targets
    y_true = X @ w_true
    noise = torch.randn(num_samples, 1) * noise_std
    y = y_true + noise

    return X, y, w_true

def generate_logistic_data(num_samples, num_features, seed=None):
    """Generates synthetic data for logistic regression.

    Args:
        num_samples (int): Number of data points.
        num_features (int): Number of features.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (X, y, w_true)
            X (torch.Tensor): Feature matrix (num_samples, num_features).
            y (torch.Tensor): Target vector (0 or 1) (num_samples, 1).
            w_true (torch.Tensor): True weight vector generating the logits (num_features, 1).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate true weights
    w_true = torch.randn(num_features, 1)

    # Generate features
    X = torch.randn(num_samples, num_features)

    # Generate logits and probabilities
    logits = X @ w_true
    probabilities = torch.sigmoid(logits)

    # Generate binary labels based on probabilities
    y = torch.bernoulli(probabilities).float()

    return X, y, w_true 