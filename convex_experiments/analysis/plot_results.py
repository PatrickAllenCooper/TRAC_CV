import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse
import numpy as np

# Define consistent colors for optimizers
OPTIMIZER_COLORS = {
    'SGD': '#1f77b4', 
    'Adam': '#ff7f0e',
    'Adagrad': '#2ca02c',
    'RMSprop': '#d62728',
    'TRAC_SGD': '#9467bd', 
    'TRAC_Adam': '#8c564b',
}

# Metrics to plot (y-axis) and their preferred y-axis labels
METRICS_TO_PLOT = {
    'loss': 'Loss',
    'gradient_norm': 'Gradient Norm',
    'distance_sq_to_opt': 'Squared Distance to Optimum',
    'trac_s_sum': 'TRAC Sum(s)',
}

# X-axis options
X_AXIS_OPTIONS = ['iteration', 'time']

def plot_results(log_dir, task, optimizers, seeds, x_axis, output_dir):
    """Loads results from CSV files and generates plots."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_data = []
    for optimizer in optimizers:
        for seed in seeds:
            file_pattern = os.path.join(log_dir, f"results_{task}_{optimizer}_seed{seed}.csv")
            matching_files = glob.glob(file_pattern)
            if not matching_files:
                print(f"Warning: No file found for {optimizer}, seed {seed}, task {task}. Pattern: {file_pattern}")
                continue
            
            try:
                df = pd.read_csv(matching_files[0])
                df['optimizer'] = optimizer
                df['seed'] = seed
                all_data.append(df)
            except Exception as e:
                print(f"Error reading file {matching_files[0]}: {e}")

    if not all_data:
        print("No data loaded. Exiting.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Set seaborn style
    sns.set_theme(style="darkgrid")

    # Plot each metric
    for metric, y_label in METRICS_TO_PLOT.items():
        if metric not in combined_df.columns:
            print(f"Metric '{metric}' not found in data, skipping plot.")
            continue

        plt.figure(figsize=(10, 6))
        
        # Use lineplot which handles aggregation over seeds automatically (mean + confidence interval)
        sns.lineplot(
            data=combined_df,
            x=x_axis,
            y=metric,
            hue='optimizer',
            palette=OPTIMIZER_COLORS,
            # estimator='mean', # default
            # errorbar=('ci', 95), # default
            legend='full'
        )

        plt.title(f'{task.capitalize()} Task: {y_label} vs. {x_axis.capitalize()}')
        plt.xlabel(x_axis.capitalize())
        plt.ylabel(y_label)
        plt.legend(title='Optimizer')
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(output_dir, f"plot_{task}_{metric}_vs_{x_axis}.png")
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        plt.close() # Close the figure to free memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from convex optimization experiments.")
    parser.add_argument('--log_dir', type=str, default='../logs', help='Directory containing the result CSV files.')
    parser.add_argument('--task', type=str, required=True, choices=['linear', 'logistic'], help='Task name to plot results for.')
    parser.add_argument('--optimizers', nargs='+', default=['SGD', 'Adam', 'Adagrad', 'RMSprop', 'TRAC_SGD', 'TRAC_Adam'], 
                        help='List of optimizers to include in the plot.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42], help='List of random seeds used in the experiments.')
    parser.add_argument('--x_axis', type=str, default='iteration', choices=X_AXIS_OPTIONS, 
                        help='Variable for the x-axis (iteration or time).')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save the generated plots.')

    args = parser.parse_args()

    # Adjust log_dir relative to this script's location if using default
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if args.log_dir == '../logs': # Default value check
        args.log_dir = os.path.join(script_dir, args.log_dir)
    if args.output_dir == 'plots': # Default value check
         args.output_dir = os.path.join(script_dir, args.output_dir)

    plot_results(args.log_dir, args.task, args.optimizers, args.seeds, args.x_axis, args.output_dir) 