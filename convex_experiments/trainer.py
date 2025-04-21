import torch
import time
import numpy as np
from typing import Dict, Any, Callable, List, Optional

def train_convex(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    num_iterations: int,
    w_true: Optional[torch.Tensor] = None,
    log_interval: int = 10,
) -> List[Dict[str, Any]]:
    """Trains a model on a convex task and logs metrics.

    Args:
        model: The PyTorch model to train.
        X: Input features (torch.Tensor).
        y: Target values (torch.Tensor).
        loss_fn: The loss function.
        optimizer: The optimizer instance.
        num_iterations: Total number of optimization steps.
        w_true: Optional true weight vector for distance calculation.
        log_interval: How often to record metrics (every N iterations).

    Returns:
        A list of dictionaries, where each dictionary contains the metrics
        for a logged iteration.
    """
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device)
    if w_true is not None:
        w_true = w_true.to(device)

    log_data = []
    start_time = time.time()

    for iteration in range(num_iterations):
        model.train() # Ensure model is in training mode

        # --- Forward pass ---
        outputs = model(X)
        loss = loss_fn(outputs, y)

        # --- Backward pass and Optimization ---
        optimizer.zero_grad()
        loss.backward()

        # --- Calculate Gradient Norm (before optimizer step) ---
        total_grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm_sq += p.grad.data.norm(2).item() ** 2
        gradient_norm = np.sqrt(total_grad_norm_sq)

        optimizer.step()

        # --- Logging --- 
        if iteration % log_interval == 0 or iteration == num_iterations - 1:
            elapsed_time = time.time() - start_time
            current_loss = loss.item()

            metrics = {
                'iteration': iteration,
                'time': elapsed_time,
                'loss': current_loss,
                'gradient_norm': gradient_norm,
            }

            # Optional: Distance to true parameters
            if w_true is not None:
                with torch.no_grad():
                    # Assuming the model is a simple linear layer
                    current_w = next(model.parameters()).data.clone().view_as(w_true)
                    dist_sq = torch.sum((current_w - w_true) ** 2).item()
                    metrics['distance_sq_to_opt'] = dist_sq

            # Optional: TRAC specific logging
            if hasattr(optimizer, 'state') and '_trac' in optimizer.state:
                trac_state = optimizer.state['_trac']
                if 's' in trac_state:
                     metrics['trac_s_sum'] = torch.sum(trac_state['s']).item()

            log_data.append(metrics)

            # Print progress (optional)
            # print(f"Iter: {iteration:5d} | Loss: {current_loss:.4f} | GradNorm: {gradient_norm:.4f} | Time: {elapsed_time:.2f}s")

    return log_data 