# MCMC Implementation

This repository contains implementations of Metropolis-Hastings and Hamiltonian Monte Carlo (HMC) samplers, along with several common probability distributions.

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

## Usage

The main script can be run with various command-line arguments. Basic usage:

```bash
python main.py [options]
```

### Default Parameters

When running `python main.py` without any arguments, the following defaults are used:

```python
--distribution="normal"     # Target distribution to sample from
--dim=1                    # Dimension of the probability space
--sampler="metropolis"     # MCMC sampler to use
--iterations=10000         # Number of MCMC iterations
--step_size=0.1           # Step size for HMC
--scale_proposal=0.1      # Scale for Metropolis proposal distribution
--L=50                    # Number of leapfrog steps (HMC only)
```

### Command-line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--distribution` | Target distribution | `"normal"` | `"normal"`, `"banana"`, `"mixture"` |
| `--dim` | Dimension of probability space | `1` | Any positive integer |
| `--sampler` | MCMC sampler type | `"metropolis"` | `"metropolis"`, `"hmc"` |
| `--iterations` | Number of iterations | `10000` | Any positive integer |
| `--step_size` | Step size for HMC | `0.1` | Any positive float |
| `--scale_proposal` | Scale for Metropolis proposal | `0.1` | Any positive float |
| `--L` | Number of leapfrog steps (HMC) | `50` | Any positive integer |
| `--visualize` | Enable visualization | `False` | Flag (no value needed) |
| `--save_plots` | Save plots to files | `False` | Flag (no value needed) |

### Example Commands

1. Basic Metropolis sampling from normal distribution:
```bash
python main.py
```

2. HMC sampling from 2D normal distribution with visualization:
```bash
python main.py --sampler hmc --dim 2 --visualize
```

3. Metropolis sampling from mixture distribution with custom proposal:
```bash
python main.py --distribution mixture --scale_proposal 0.5 --iterations 5000
```

4. HMC sampling from banana distribution:
```bash
python main.py --distribution banana --sampler hmc --step_size 0.1 --L 20
```

## Available Distributions

1. **Normal Distribution** (`normal`)
   - Multivariate normal distribution
   - Supports any dimension
   - Parameters controlled by `--dim`

2. **Banana Distribution** (`banana`)
   - Banana-shaped distribution (common MCMC test case)
   - Only supports 2D
   - Automatically sets `dim=2`

3. **Mixture Distribution** (`mixture`)
   - Mixture of Gaussian distributions
   - Supports any dimension
   - Default: 2 components with random means

## Visualization

When `--visualize` is enabled:
- 1D distributions: Histogram of samples with target density overlay
- 2D distributions: Scatter plot of samples with density contours
- All dimensions: Trace plots showing sample evolution

Use `--save_plots` to save the visualizations to files.

## Notes

- The Banana distribution only works with `dim=2`
- HMC requires target distributions with implemented gradient
- For high dimensions, consider using HMC over Metropolis
- Adjust `scale_proposal` for Metropolis or `step_size` for HMC if acceptance rate is too low/high 