# MCMC Implementation

This repository contains implementations of Metropolis-Hastings and Hamiltonian Monte Carlo (HMC) samplers, along with several common probability distributions.

## Installation

```bash
# Clone the repository
git clone https://github.com/AlexMan2000/DS-GA-1005-Project

# Install dependencies
cd codes
conda env create -f environment.yml --prefix ./venv
conda activate ./venv
pip install -r requirements.txt
```

## Usage
### Main Script
The main script can be run with various command-line arguments. Basic usage:

```bash
python main.py [options]
```

#### Default Parameters

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

#### Command-line Arguments

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

#### Example Commands

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

### Experiment Scripts

#### 1. Donut Distribution Comparison (`experiments/donut_comparison.py`)

This script replicates the visualization from [T. Begley's blog post](https://www.tcbegley.com/blog/posts/mcmc-part-2), comparing Random-Walk Metropolis (with two different scales) and HMC on a donut-shaped distribution (n-dimensional shell).

**Default Parameters:**
```python
--dim=2                    # Dimension of the probability space
--n-samples=1000          # Number of samples to generate
--hmc-step-size=0.1       # Step size for HMC
--hmc-leapfrog-steps=50   # Number of leapfrog steps for HMC
--output="donut_comparison_2d.png"  # Output file path
```

**Running the script:**
```bash
python experiments/donut_comparison.py
```
(Generates `donut_comparison_2d.png` in the `codes` directory by default)

**Examples:**

*   **Run with 4 dimensions:**
    ```bash
    python experiments/donut_comparison.py --dim 4
    ```
    (Generates `donut_comparison_4d.png`)

*   **Custom parameters (3D, 500 samples, custom HMC settings):**
    ```bash
    python experiments/donut_comparison.py --dim 3 --n-samples 500 --hmc-step-size 0.05 --hmc-leapfrog-steps 100
    ```
    (Generates `donut_comparison_3d.png`)

*   **Custom output file (relative to `codes` directory):**
    ```bash
    python experiments/donut_comparison.py --dim 5 --output experiments/my_donut_comparison.png
    ```

Use `python experiments/donut_comparison.py --help` for all options.

#### 2. High-Dimensional Gaussian Comparison (`experiments/gaussian_comparison.py`)

This script compares the performance (Effective Sample Size, Acceptance Rate, Time) of Metropolis and HMC when sampling from a high-dimensional standard Gaussian distribution.

**Default Parameters:**
```python
--dimensions=[2, 5, 10, 20, 50, 100]  # Dimensions to test
--n-samples=1000                      # Number of samples per dimension
--n-warmup=1000                       # Number of warmup samples
--hmc-step-size=0.1                   # Step size for HMC
--hmc-leapfrog-steps=50               # Number of leapfrog steps for HMC
--metropolis-scale=0.1                # Scale for Metropolis proposal
--output="dimension_comparison.png"    # Output file path
```

**Running the script:**
```bash
python experiments/gaussian_comparison.py
```

This will:
*   Run the comparison for dimensions [2, 5, 10, 20, 50, 100]
*   Print summary tables to the console showing:
    - Effective Sample Size (ESS) for each dimension
    - Acceptance rates for both samplers
    - Time taken per sample
*   Save comparison plots to `dimension_comparison.png` in the `codes` directory

**Examples:**

*   **Test specific dimensions:**
    ```bash
    python experiments/gaussian_comparison.py --dimensions 2 5 10
    ```

*   **Custom sample size and warmup:**
    ```bash
    python experiments/gaussian_comparison.py --n-samples 2000 --n-warmup 500
    ```

*   **Custom HMC parameters:**
    ```bash
    python experiments/gaussian_comparison.py --hmc-step-size 0.05 --hmc-leapfrog-steps 100
    ```

*   **Custom Metropolis scale:**
    ```bash
    python experiments/gaussian_comparison.py --metropolis-scale 0.2
    ```

Use `python experiments/gaussian_comparison.py --help` for all options.

## Available Distributions

1. **Normal Distribution** (`normal`)
   - Multivariate normal distribution
   - Supports any dimension
   - Parameters controlled by `--dim`

2. **Banana Distribution** (`banana`)
   - Banana-shaped distribution (common MCMC test case)
   - Only supports 2D
   - Need to manually set to `--dim 2`

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