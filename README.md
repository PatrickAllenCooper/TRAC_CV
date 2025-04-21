# TRAC: Adaptive Parameter-free Optimization ⚡️
[![Open In Colab](https://img.shields.io/badge/project%20page-purple?logo=GitHub&logoColor=white
)](https://computationalrobotics.seas.harvard.edu/TRAC/)[![arXiv](https://img.shields.io/badge/arXiv-2405.16642-b31b1b.svg)](https://arxiv.org/abs/2405.16642) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c5OxMa5fiSVnl5w6J7flrjNUteUkp6BV?usp=sharing)[![Pypi](https://img.shields.io/badge/trac__optimizer-optimizer?logo=python&label=pip%20install&labelColor=white&color=gray&link=https%3A%2F%2Fpypi.org%2Fproject%2Ftrac-optimizer%2F)](https://pypi.org/project/trac-optimizer/)



This repository is the official implementation of the **TRAC** optimizer in ***Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning***.

How can you _quickly_ adapt to new tasks or distribution shifts? Without knowing when or how much to adapt? And without _ANY_ tuning? 
 🤔💭

Well, we suggest you get on the fast **TRAC** 🏎️💨.

**TRAC** is a parameter-free optimizer for continual environments inspired by [online convex optimization](https://arxiv.org/abs/1912.13213) and uses [discounted adaptive online prediction](https://arxiv.org/abs/2402.02720).

**Update [08/20/24]**: _TRAC is now supported for JAX and Optax!_

## Implement with only one line change with both PyTorch/JAX!
Like other [meta-tuners](https://openreview.net/pdf?id=uhKtQMn21D), TRAC can work with any of your continual, fine-tuning, or lifelong experiments with just one line change.
```python
pip install trac-optimizer
```
**PyTorch**
```python
from trac_optimizer import start_trac
# original optimizer
optimizer = torch.optim.Adam
lr = 0.001
optimizer = start_trac(log_file='logs/trac.text', optimizer)(model.parameters(), lr=lr)
```
**JAX**
```python
from trac_optimizer.experimental.jax.trac import start_trac
# original optimizer
optimizer = optax.adam(1e-3)
optimizer = start_trac(optimizer)
```

After this modification, you can continue using your optimizer methods exactly as you did before. Whether it's calling `optimizer.step()` to update your model's parameters or `optimizer.zero_grad()` to clear gradients, everything stays the same. TRAC integrates into your existing workflow without any additional overhead.

## Control Experiments

We recommend running ``main.ipynb`` in Google Colab. This approach requires no setup, making it easy to get started with our control experiments. If you run locally, to install the necessary dependencies, simply:

```setup
pip install -r requirements.txt
```
![Control Experiment](figures/control.png)


## Vision-based RL Experiments

Our vision-based experiments for [Procgen](https://openai.com/index/procgen-benchmark/) and [Atari](https://www.gymlibrary.dev/environments/atari/index.html) are hosted in the `vision_exp` directory, which is based off [this Procgen Pytorch implementation](https://github.com/joonleesky/train-procgen-pytorch). 

To initiate an experiment with the default configuration in the Procgen "starpilot" environment, use the command below. You can easily switch to other game environments, like Atari, by altering the `--exp_name="atari"` parameter:

```bash
python vision_exp/train.py --exp_name="procgen" --env_name="starpilot" --optimizer="TRAC" --warmstart_step=0
```
<p align="center">
  <img src="figures/games1.gif" alt="Game 1" width="30%">
  <img src="figures/games2.gif" alt="Game 2" width="30%">
  <img src="figures/games3.gif" alt="Game 3" width="30%">
</p>


## Convex Optimization Experiments

This repository also includes a framework for comparing TRAC against standard optimizers (SGD, Adam, Adagrad, RMSprop) on simple convex optimization tasks (Linear and Logistic Regression using synthetic data). This allows for analyzing the convergence properties of TRAC in a controlled setting.

### Setup

First, ensure you have the necessary dependencies installed, including those for plotting:

```bash
pip install -r requirements.txt
```

### Running Experiments

The main script for running these experiments is `convex_experiments/main_convex.py`. You can run it from the command line, specifying the task, optimizer, and other parameters.

Example commands (run from the root directory):

```bash
# Run Linear Regression with Adam (learning rate 0.001, seed 1)
python convex_experiments/main_convex.py --task linear --optimizer Adam --lr 0.001 --seed 1

# Run Linear Regression with TRAC-Adam (seed 1)
python convex_experiments/main_convex.py --task linear --optimizer TRAC_Adam --seed 1

# Run Logistic Regression with SGD (learning rate 0.1, seed 2)
python convex_experiments/main_convex.py --task logistic --optimizer SGD --lr 0.1 --seed 2

# Run Logistic Regression with TRAC-SGD (seed 2)
python convex_experiments/main_convex.py --task logistic --optimizer TRAC_SGD --seed 2 
```

Results (metrics per iteration) will be saved as CSV files in the `convex_experiments/logs/` directory.

*Note: For a fair comparison, you may need to tune the learning rate (`--lr`) for the baseline optimizers (SGD, Adam, etc.) on each specific task.*

### Plotting Results

After running experiments for different optimizers and seeds, you can generate comparison plots using the `convex_experiments/analysis/plot_results.py` script.

Example command (run from the root directory):

```bash
# Plot results for the linear task, comparing Adam and TRAC_Adam 
# using seeds 1 and 2. Plot against iteration number.
python convex_experiments/analysis/plot_results.py \
    --task linear \
    --optimizers Adam TRAC_Adam \
    --seeds 1 2 \
    --x_axis iteration \
    --log_dir convex_experiments/logs \
    --output_dir convex_experiments/analysis/plots
```

This will generate plots (e.g., Loss vs. Iteration, Gradient Norm vs. Iteration) comparing the specified optimizers, averaged over the provided seeds, and save them in the `convex_experiments/analysis/plots/` directory.
