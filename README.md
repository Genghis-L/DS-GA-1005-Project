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

#### Available Distributions

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

#### Visualization

When `--visualize` is enabled:
- 1D distributions: Histogram of samples with target density overlay
- 2D distributions: Scatter plot of samples with density contours
- All dimensions: Trace plots showing sample evolution

Use `--save_plots` to save the visualizations to files.

#### Notes

- The Banana distribution only works with `dim=2`
- HMC requires target distributions with implemented gradient
- For high dimensions, consider using HMC over Metropolis
- Adjust `scale_proposal` for Metropolis or `step_size` for HMC if acceptance rate is too low/high 

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

#### 2. Proposal Distribution Comparison (`experiments/proposal_comparison.py`)

This script compares different proposal distributions for the Metropolis sampler, including Normal, Student's t, and Uniform proposals. It analyzes acceptance rates, ESS (Effective Sample Size), computational efficiency, and mixing behavior through trajectory plots.

**Default Parameters:**
```python
--dim=1                    # Dimension of the probability space
--n-samples=10000         # Number of samples to generate
--target-std=1.0          # Standard deviation of target normal distribution
--proposal-types=['normal', 'student-t', 'uniform']  # Proposal types to compare
--scales=[0.1, 0.5, 1.0, 2.0, 5.0]  # Scales to test for each proposal
--output="proposal_comparison_1d.png"  # Output file prefix
```

**Running the script:**
```bash
python experiments/proposal_comparison.py
```

This will generate:
1. Trajectory plots for each scale value:
   - Short run plots (first 200 iterations): `proposal_comparison_1d_trajectory_short_scale{scale}.png`
   - Long run plots (first 1000 iterations): `proposal_comparison_1d_trajectory_long_scale{scale}.png`
   - One subplot per proposal type
   - First coordinate's position over time
   - Matching y-axis limits for fair comparison
2. Performance metric plots (`proposal_comparison_1d.png`):
   - Acceptance rate vs scale
   - Computation time vs scale
   - ESS vs scale
   - ESS per second vs scale

**Examples:**

*   **Compare specific proposal types:**
    ```bash
    python experiments/proposal_comparison.py --proposal-types normal student-t
    ```

*   **Custom scales and dimensions:**
    ```bash
    python experiments/proposal_comparison.py --dim 2 --scales 0.1 0.5 1.0
    ```

*   **Adjust target distribution:**
    ```bash
    python experiments/proposal_comparison.py --target-std 2.0
    ```

*   **Custom output prefix:**
    ```bash
    python experiments/proposal_comparison.py --output "my_proposal_comparison"
    ```
    This will generate:
    - `my_proposal_comparison_trajectory_short_scale{scale}.png`
    - `my_proposal_comparison_trajectory_long_scale{scale}.png`
    - `my_proposal_comparison.png`

Use `python experiments/proposal_comparison.py --help` for all options.

#### 3. HMC Integrator Comparison (`experiments/integrator_comparison.py`)

This script compares different numerical integrators for HMC: Euler, Modified Euler, and Leapfrog. It analyzes acceptance rates, energy conservation, and computational efficiency across different target distributions.

**Default Parameters:**
```python
--distribution="normal"    # Target distribution to sample from
--dim=2                   # Dimension of the probability space
--n-samples=1000          # Number of samples to generate
--target-std=1.0          # Standard deviation of target normal distribution
--donut-radius=3.0        # Target radius for donut distribution
--donut-sigma2=0.05       # Shell thickness for donut distribution
--step-sizes=[0.1, 0.5, 1.0]  # Step sizes to test
--n-leapfrog=20          # Number of integration steps
--output="integrator_comparison_{dist}_{dim}d.png"  # Output file path
```

**Examples:**

*   **Compare on normal distribution (default):**
    ```bash
    python experiments/integrator_comparison.py
    ```

*   **Compare on donut distribution:**
    ```bash
    python experiments/integrator_comparison.py --distribution donut
    ```

*   **Custom donut parameters:**
    ```bash
    python experiments/integrator_comparison.py --distribution donut --donut-radius 2.0 --donut-sigma2 0.1
    ```

*   **Compare in higher dimensions:**
    ```bash
    python experiments/integrator_comparison.py --dim 5
    ```

*   **Custom step sizes and more integration steps:**
    ```bash
    python experiments/integrator_comparison.py --step-sizes 0.01 0.05 0.1 --n-leapfrog 50
    ```

The script generates two types of plots:
1. `*_metrics.png`: Shows comparison of:
   - Acceptance rates vs step size
   - Computation time vs step size
   - Energy conservation error vs step size
2. `*_samples.png` (for 2D only): Shows:
   - Sample distributions for each integrator
   - Target density contours
   - Acceptance rates and energy errors
   - One row per step size, one column per integrator

#### 4. High-Dimensional Gaussian Comparison (`experiments/gaussian_comparison.py`)

This script compares the performance (Effective Sample Size, Acceptance Rate, Time) of Metropolis and HMC when sampling from a high-dimensional standard Gaussian distribution. It also generates trajectory plots comparing the mixing behavior of both samplers.

**Default Parameters:**
```python
--dimensions=[2, 5, 10, 20, 50, 100]  # Dimensions to test
--n-samples=1000                      # Number of samples per dimension
--n-warmup=1000                       # Number of warmup samples
--hmc-step-size=0.1                   # Step size for HMC
--hmc-leapfrog-steps=50              # Number of leapfrog steps for HMC
--metropolis-scale=0.1               # Scale for Metropolis proposal
--output="gaussian_comparison"        # Output file prefix
```

**Running the script:**
```bash
python experiments/gaussian_comparison.py
```

This will:
*   Run the comparison for dimensions [2, 5, 10, 20, 50, 100]
*   Generate three types of plots:
    1. Short trajectory plots (first 200 iterations) for 2D and 100D cases
    2. Long trajectory plots (first 1000 iterations) for 2D and 100D cases
    3. Performance metric plots comparing:
       - Effective Sample Size (ESS) vs dimension
       - Computation time vs dimension
       - Acceptance rate vs dimension
       - ESS per second vs dimension
*   Print summary tables to the console showing:
    - ESS for each dimension
    - Acceptance rates for both samplers
    - Time taken per sample

The trajectory plots show:
- Left panel: Random-walk Metropolis trajectory
- Right panel: HMC trajectory
- First coordinate's position over iterations
- Black dots for sample points
- Matching y-axis limits for fair comparison

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

*   **Custom output prefix:**
    ```bash
    python experiments/gaussian_comparison.py --output "my_gaussian_comparison"
    ```
    This will generate:
    - `my_gaussian_comparison_trajectory_short.png`
    - `my_gaussian_comparison_trajectory_long.png`
    - `my_gaussian_comparison_metrics.png`

Use `python experiments/gaussian_comparison.py --help` for all options.

#### 5. Hamiltonian Dynamics Visualization (`experiments/integrator_trajectory.py`)

This script visualizes how different numerical integrators (Euler, Modified Euler, and Leapfrog) approximate Hamiltonian dynamics for a simple harmonic oscillator system. It recreates the phase space trajectories similar to those shown in Neal's HMC paper.

**Default Parameters:**
```python
--step-size=0.3           # Base step size for integration
--n-steps=20              # Number of integration steps
--q0=0.0                  # Initial position
--p0=1.0                  # Initial momentum
--output="integrator_trajectories.png"  # Output file path
```

The script generates a 2x2 grid of plots showing:
1. Euler's method with stepsize 0.3
2. Modified Euler's method with stepsize 0.3
3. Leapfrog method with stepsize 0.3
4. Leapfrog method with stepsize 1.2

Each plot shows:
- The true trajectory (gray circle)
- The numerical approximation (black dots and lines)
- Position (q) vs Momentum (p) in phase space

**Examples:**

*   **Default visualization:**
    ```bash
    python experiments/integrator_trajectory.py
    ```

*   **Custom step size and initial conditions:**
    ```bash
    python experiments/integrator_trajectory.py --step-size 0.2 --q0 0.5 --p0 0.5
    ```

*   **More integration steps:**
    ```bash
    python experiments/integrator_trajectory.py --n-steps 50
    ```

The visualization demonstrates key properties of each integrator:
- Euler's method: Energy tends to increase (spiral outward)
- Modified Euler's method: Better energy conservation
- Leapfrog method: Excellent energy conservation even with larger step sizes

#### 6. Sample Visualization Comparison

Compare different sampling methods on 2D distributions:

```bash
# Compare all methods on donut distribution
python experiments/sample_visualization_comparison.py

# Use banana distribution with custom parameters
python experiments/sample_visualization_comparison.py \
    --distribution banana \
    --n_samples 10000 \
    --metropolis_scales 0.1 0.3 0.5 \
    --hmc_step_sizes 0.05 0.1 \
    --hmc_n_steps 30 50
```

The script produces:
- Sample scatter plots with target density contours
- Acceptance rates for each method
- Effective Sample Size (ESS)
- ESS per second (sampling efficiency)

Default parameters:
- `n_samples`: 5000
- `metropolis_scales`: [0.1, 0.5, 1.0]
- `hmc_step_sizes`: [0.1]
- `hmc_n_steps`: [50]

