import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
import argparse
import random
import numpy as np

# Import local modules
from models.linear_models import LinearRegressionModel, LogisticRegressionModel
from data.synthetic_data import generate_linear_data, generate_logistic_data
from optimizers.trac_pytorch import start_trac
from trainer import train_convex

def main(args):
    # --- Setup ---
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data --- 
    w_true = None
    if args.task == 'linear':
        X, y, w_true = generate_linear_data(args.num_samples, args.num_features, args.noise_std, args.seed)
        loss_fn = nn.MSELoss()
        model = LinearRegressionModel(args.num_features).to(device)
    elif args.task == 'logistic':
        X, y, w_true = generate_logistic_data(args.num_samples, args.num_features, args.seed)
        loss_fn = nn.BCEWithLogitsLoss()
        model = LogisticRegressionModel(args.num_features).to(device)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # --- Optimizer --- 
    # TRAC requires a log file, create a dummy one for now
    trac_dummy_log = os.path.join(args.log_dir, f"trac_internal_{args.optimizer}_{args.task}_{args.seed}.log")
    if os.path.exists(trac_dummy_log):
        os.remove(trac_dummy_log) # Ensure clean start if re-running

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'TRAC_SGD':
        optimizer_class = start_trac(log_file=trac_dummy_log, Base=optim.SGD)
        optimizer = optimizer_class(model.parameters(), lr=args.lr) # Pass lr, though TRAC might ignore/adapt it
    elif args.optimizer == 'TRAC_Adam':
        optimizer_class = start_trac(log_file=trac_dummy_log, Base=optim.Adam)
        optimizer = optimizer_class(model.parameters(), lr=args.lr) # Pass lr
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    print(f"Starting training for task: {args.task}, optimizer: {args.optimizer}, seed: {args.seed}")

    # --- Training --- 
    log_data = train_convex(
        model=model,
        X=X,
        y=y,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_iterations=args.num_iterations,
        w_true=w_true, # Pass w_true for distance calculation
        log_interval=args.log_interval
    )

    # --- Save Results --- 
    log_df = pd.DataFrame(log_data)
    results_filename = os.path.join(args.log_dir, f"results_{args.task}_{args.optimizer}_seed{args.seed}.csv")
    log_df.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run convex optimization experiments.")
    parser.add_argument('--task', type=str, required=True, choices=['linear', 'logistic'], help='Optimization task')
    parser.add_argument('--optimizer', type=str, required=True, 
                        choices=['SGD', 'Adam', 'Adagrad', 'RMSprop', 'TRAC_SGD', 'TRAC_Adam'], 
                        help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for standard optimizers')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of data samples')
    parser.add_argument('--num_features', type=int, default=20, help='Number of features')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Noise std deviation for linear regression')
    parser.add_argument('--num_iterations', type=int, default=500, help='Number of training iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='convex_experiments/logs', help='Directory to save logs and results')
    parser.add_argument('--log_interval', type=int, default=1, help='Log metrics every N iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    main(args) 