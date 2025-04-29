"""
This experiment is to compare the performance of different HMC sampling mechanisms under 100d for 
standard gaussian distribution and donut distribution.

The proposal for HMC should fix to be gaussian.

The step size for HMC with metropolis-hasting correction should be fixed to be 0.1
The step size for HMC without metropolis-hasting correction should be [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]

This will generate four figures 2x3:
For the first line of figures, there should be 3 figures, the title should be "Target Distribution: Standard Gaussian comparison\n Number of points to sample: {sample_size}",
In the figure (1, 1), 
The x-axis should be the step size of the HMC algorithm and the y-axis should be the ESS of the algorithm.
There should be two lines, one for HMC with metropolis-hasting correction(independent of the x-axis step size) and one for HMC without metropolis-hasting correction, following the x-axis step size.

In the figure (1, 2),
The x-axis should be the step size of the algorithm(for HMC, it should be the step size of the leapfrog integrator, EPSILON,  for MCMC, it should be the step size of the proposal), and the y-axis should be the Computational Time of the algorithm.
There should be two lines, one for HMC with metropolis-hasting correction(independent of the x-axis step size) and one for HMC without metropolis-hasting correction, following the x-axis step size.


In the figure (1, 3),
The x-axis should be the step size of the algorithm(for HMC, it should be the step size of the leapfrog integrator, EPSILON,  for MCMC, it should be the step size of the proposal), and the y-axis should be the Energy Error of the algorithm.
There should be two lines, one for HMC with metropolis-hasting correction(independent of the x-axis step size) and one for HMC without metropolis-hasting correction, following the x-axis step size.


For the second line of figures, there should be 3 figures, the title should be "Target Distribution: Donut Distribution comparison\n Number of points to sample: {sample_size}",
Each figure in the line should show the same stuff(like the x-axis, y-axis, and the lines, though the stats varies) as the first line.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List
import sys
import os

# Add the parent directory to the path to import from algos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from algos import HMCWithIntegrator, compute_energy_error, compute_relative_energy_error


# Standard Gaussian target
def standard_gaussian(dim):
    class StandardGaussian:
        def __init__(self, dim):
            self.dim = dim
        def __call__(self, x):
            return np.exp(-0.5 * np.sum(x * x)) / (2 * np.pi) ** (self.dim / 2)
        def log_density(self, x):
            return -0.5 * np.sum(x * x) - (self.dim / 2) * np.log(2 * np.pi)
        def grad_log_density(self, x):
            return -x
    return StandardGaussian(dim)

# Donut target
def donut_distribution(dim, radius=3.0, sigma2=0.05):
    class Donut:
        def __init__(self, dim, radius, sigma2):
            self.dim = dim
            self.radius = radius
            self.sigma2 = sigma2
        def __call__(self, x):
            r = np.linalg.norm(x)
            return np.exp(-(r - self.radius) ** 2 / self.sigma2)
        def log_density(self, x):
            r = np.linalg.norm(x)
            return -(r - self.radius) ** 2 / self.sigma2
        def grad_log_density(self, x):
            r = np.linalg.norm(x)
            if r == 0:
                return np.zeros_like(x)
            return 2 * x * (self.radius / r - 1) / self.sigma2
    return Donut(dim, radius, sigma2)

def run_experiment(target_class, dim=100, n_samples=1000, n_warmup=100, L=50):
    # Step sizes
    step_sizes_no_metro = [0.002, 0.005, 0.01, 0.025, 0.04, 0.055, 0.07, 0.085, 0.1, 0.2]
    step_size_metro = 0.1
    # Results
    results = {
        'step_sizes': step_sizes_no_metro,
        'ess_no_metro': [],
        'ess_metro': None,
        'time_no_metro': [],
        'time_metro': None,
        'relative_energy_no_metro': [],
        'relative_energy_metro': None
    }
    # HMC with Metropolis (fixed step size)
    target = target_class(dim)
    initial = lambda: np.zeros(dim)
    start = time.time()
    sampler_metro = HMCWithIntegrator(
        target=target,
        initial=initial,
        iterations=n_samples + n_warmup,
        L=L,
        step_size=step_size_metro,
        kinetic_energy='gaussian',
        use_metropolis=True
    )
    samples_metro = np.array(sampler_metro.run()[n_warmup:])
    end = time.time()
    results['ess_metro'] = np.mean([compute_ess(samples_metro[:, i]) for i in range(dim)])
    results['time_metro'] = end - start
    # results['energy_metro'] = compute_energy_error(sampler_metro)
    results['relative_energy_metro'] = compute_relative_energy_error(sampler_metro)
    # HMC without Metropolis (varying step size)
    for step_size in step_sizes_no_metro:
        sampler_no_metro = HMCWithIntegrator(
            target=target,
            initial=initial,
            iterations=n_samples + n_warmup,
            L=L,
            step_size=step_size,
            kinetic_energy='gaussian',
            use_metropolis=False
        )
        start = time.time()
        samples_no_metro = np.array(sampler_no_metro.run()[n_warmup:])
        end = time.time()
        ess = np.mean([compute_ess(samples_no_metro[:, i]) for i in range(dim)])
        results['ess_no_metro'].append(ess)
        results['time_no_metro'].append(end - start)
        # results['energy_no_metro'].append(compute_energy_error(sampler_no_metro))
        results['relative_energy_no_metro'].append(compute_relative_energy_error(sampler_no_metro))
    return results

def compute_ess(samples: np.ndarray) -> float:
    n = len(samples)
    if n <= 1:
        return 0.0
    max_lag = min(50, n//3)
    mean = np.mean(samples)
    var = np.var(samples)
    if var == 0 or not np.isfinite(var):
        return 0.0
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag >= n:
            break
        c0 = samples[:-lag] - mean if lag > 0 else samples - mean
        c1 = samples[lag:] - mean
        if len(c0) == 0 or len(c1) == 0:
            break
        acf[lag] = np.mean(c0 * c1) / var
    cutoff = np.where((acf < 0.05) | (acf < 0))[0]
    if len(cutoff) > 0:
        max_lag = cutoff[0]
    ess = n / (1 + 2 * np.sum(acf[:max_lag]))
    return max(1.0, ess)

def plot_experiment_single(results: Dict, target_name: str, sample_size: int, save_path: str):
    step_sizes = results['step_sizes']
    step_size_metro = 0.1
    L = 50
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"HMC Step Size Comparison (100d) - {target_name}\n"
        f"Number of points to sample: {sample_size}\n"
        f"HMC with Metropolis step size (epsilon): {step_size_metro}, L: {L}, with leapfrog integrator\n"
        f"HMC w/o Metropolis step sizes: {step_sizes}"
    )
    fig.subplots_adjust(top=0.90)

    # ESS
    axes[0, 0].plot(step_sizes, results['ess_no_metro'], marker='o', label='HMC w/o Metropolis')
    axes[0, 0].hlines(results['ess_metro'], step_sizes[0], step_sizes[-1], colors='r', linestyles='--', label='HMC w/ Metropolis')
    axes[0, 0].set_xlabel('Step size')
    axes[0, 0].set_ylabel('ESS')
    axes[0, 0].set_title('ESS')
    axes[0, 0].set_xscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # Energy Error
    axes[0, 1].plot(step_sizes, results['relative_energy_no_metro'], marker='o', label='HMC w/o Metropolis')
    axes[0, 1].hlines(results['relative_energy_metro'], step_sizes[0], step_sizes[-1], colors='r', linestyles='--', label='HMC w/ Metropolis')
    axes[0, 1].set_xlabel('Step size')
    axes[0, 1].set_ylabel('Relative Energy Error')
    axes[0, 1].set_title('Relative Energy Error')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    # Computation Time
    axes[1, 0].plot(step_sizes, results['time_no_metro'], marker='o', label='HMC w/o Metropolis')
    axes[1, 0].hlines(results['time_metro'], step_sizes[0], step_sizes[-1], colors='r', linestyles='--', label='HMC w/ Metropolis')
    axes[1, 0].set_xlabel('Step size')
    axes[1, 0].set_ylabel('Computation Time (s)')
    axes[1, 0].set_title('Computation Time')
    axes[1, 0].set_xscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    # Time per Sample
    axes[1, 1].plot(step_sizes, np.array(results['time_no_metro'])/sample_size, marker='o', label='HMC w/o Metropolis')
    axes[1, 1].hlines(results['time_metro']/sample_size, step_sizes[0], step_sizes[-1], colors='r', linestyles='--', label='HMC w/ Metropolis')
    axes[1, 1].set_xlabel('Step size')
    axes[1, 1].set_ylabel('Time per Sample (s)')
    axes[1, 1].set_title('Time per Sample')
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    dim = 500
    n_samples = 1000
    n_warmup = 100
    L = 50
    # Standard Gaussian
    results_gauss = run_experiment(standard_gaussian, dim, n_samples, n_warmup, L)
    plot_experiment_single(results_gauss, 'Standard Gaussian', n_samples, 'hmc_step_size_comparison_gaussian.png')
    # Donut
    results_donut = run_experiment(lambda d: donut_distribution(d, radius=3.0, sigma2=0.05), dim, n_samples, n_warmup, L)
    plot_experiment_single(results_donut, 'Donut Distribution', n_samples, 'hmc_step_size_comparison_donut.png')

if __name__ == "__main__":
    main()

