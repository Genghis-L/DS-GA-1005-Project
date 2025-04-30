"""
This experiment is to compare the performance of different HMC sampling mechanisms under 100d for 
standard gaussian distribution and donut distribution.

The proposal for HMC should fix to be gaussian.

The epsilon * L for HMC with metropolis-hasting correction should be fixed to be 0.5
The epsilon * L for HMC without metropolis-hasting correction should also be 0.5, but we vary epsilon

This will generate four figures 2x3:
For the first line of figures, there should be 3 figures, the title should be "Target Distribution: Standard Gaussian comparison\n Number of points to sample: {sample_size}",
In the figure (1, 1), 
The x-axis should be the step size (epsilon) of the HMC algorithm and the y-axis should be the ESS of the algorithm.
There should be two lines, one for HMC with metropolis-hasting correction(independent of the x-axis step size) and one for HMC without metropolis-hasting correction, following the x-axis step size.

In the figure (1, 2),
The x-axis should be the step size (epsilon) of the algorithm, and the y-axis should be the Computational Time of the algorithm.
There should be two lines, one for HMC with metropolis-hasting correction(independent of the x-axis step size) and one for HMC without metropolis-hasting correction, following the x-axis step size.

In the figure (1, 3),
The x-axis should be the step size (epsilon) of the algorithm, and the y-axis should be the Energy Error of the algorithm.
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
def donut_distribution(dim, radius=3.0, sigma2=0.5):
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

def run_experiment(target_class, dim=300, n_samples=1000, n_warmup=100, epsilon_L=0.5):
    # Step sizes (epsilon values)
    epsilons = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1] # without metropolis
    epsilon_metro = 0.1  # with metropolis
    
    # Results
    results = {
        'epsilons': epsilons,
        'ess_no_metro': [],
        'ess_metro': None,
        'flops_no_metro': [],
        'flops_metro': None,
        'relative_energy_no_metro': [],
        'relative_energy_metro': None
    }
    
    # HMC with Metropolis (fixed epsilon)
    target = target_class(dim)
    initial = lambda: np.zeros(dim)
    L_metro = int(epsilon_L / epsilon_metro)  # Calculate L based on epsilon_L
    
    sampler_metro = HMCWithIntegrator(
        target=target,
        initial=initial,
        iterations=n_samples + n_warmup,
        trajectory_length=epsilon_L,
        step_size=epsilon_metro,
        kinetic_energy='gaussian',
        use_metropolis=True
    )
    samples_metro = np.array(sampler_metro.run()[n_warmup:])
    results['ess_metro'] = np.mean([compute_ess(samples_metro[:, i]) for i in range(dim)])
    results['flops_metro'] = sampler_metro.get_flop_count()
    results['relative_energy_metro'] = compute_relative_energy_error(sampler_metro)
    
    # HMC without Metropolis (varying epsilon)
    for epsilon in epsilons:
        L = int(epsilon_L / epsilon)  # Calculate L based on epsilon_L
        sampler_no_metro = HMCWithIntegrator(
            target=target,
            initial=initial,
            iterations=n_samples + n_warmup,
            trajectory_length=epsilon_L,
            step_size=epsilon,
            kinetic_energy='gaussian',
            use_metropolis=False
        )
        samples_no_metro = np.array(sampler_no_metro.run()[n_warmup:])
        ess = np.mean([compute_ess(samples_no_metro[:, i]) for i in range(dim)])
        results['ess_no_metro'].append(ess)
        results['flops_no_metro'].append(sampler_no_metro.get_flop_count())
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

def plot_experiment_single(results: Dict, dim: int, target_name: str, sample_size: int, save_path: str):
    epsilons = results['epsilons']
    epsilon_metro = 0.1
    epsilon_L = 0.5
    L_metro = int(epsilon_L / epsilon_metro)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"HMC Step Size Comparison ({dim}d) - {target_name}\n"
        f"Number of points to sample: {sample_size}\n"
        f"HMC with Metropolis: epsilon={epsilon_metro}, L={L_metro}, epsilon*L={epsilon_L}\n"
        f"HMC w/o Metropolis: varying epsilon, fixed epsilon*L={epsilon_L}, epsilons: {epsilons}"
    )
    fig.subplots_adjust(top=0.90)

    # ESS
    axes[0, 0].plot(epsilons, results['ess_no_metro'], marker='o', label='HMC w/o Metropolis')
    axes[0, 0].hlines(results['ess_metro'], epsilons[0], epsilons[-1], colors='r', linestyles='--', label='HMC w/ Metropolis')
    axes[0, 0].set_xlabel('Step size (epsilon)')
    axes[0, 0].set_ylabel('ESS')
    axes[0, 0].set_title('ESS')
    axes[0, 0].set_xscale('log')
    ess_min = int(np.floor(min(min(results['ess_no_metro']), results['ess_metro'])))
    ess_max = int(np.ceil(max(max(results['ess_no_metro']), results['ess_metro'])))
    axes[0, 0].set_yticks(range(ess_min, ess_max + 1))
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Energy Error
    axes[0, 1].plot(epsilons, results['relative_energy_no_metro'], marker='o', label='HMC w/o Metropolis')
    axes[0, 1].hlines(results['relative_energy_metro'], epsilons[0], epsilons[-1], colors='r', linestyles='--', label='HMC w/ Metropolis')
    axes[0, 1].set_xlabel('Step size (epsilon)')
    axes[0, 1].set_ylabel('Relative Energy Error')
    axes[0, 1].set_title('Relative Energy Error')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # FLOPs
    axes[1, 0].plot(epsilons, results['flops_no_metro'], marker='o', label='HMC w/o Metropolis')
    axes[1, 0].hlines(results['flops_metro'], epsilons[0], epsilons[-1], colors='r', linestyles='--', label='HMC w/ Metropolis')
    axes[1, 0].set_xlabel('Step size (epsilon)')
    axes[1, 0].set_ylabel('FLOPs')
    axes[1, 0].set_title('FLOPs')
    axes[1, 0].set_xscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # ESS per FLOP
    ess_per_flop_no_metro = np.array(results['ess_no_metro']) / np.array(results['flops_no_metro'])
    ess_per_flop_metro = results['ess_metro'] / results['flops_metro']
    axes[1, 1].plot(epsilons, ess_per_flop_no_metro, marker='o', label='HMC w/o Metropolis')
    axes[1, 1].hlines(ess_per_flop_metro, epsilons[0], epsilons[-1], colors='r', linestyles='--', label='HMC w/ Metropolis')
    axes[1, 1].set_xlabel('Step size (epsilon)')
    axes[1, 1].set_ylabel('ESS per FLOP')
    axes[1, 1].set_title('ESS per FLOP')
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    dim = 300
    n_samples = 1000
    n_warmup = 100
    epsilon_L = 0.5
    
    # Standard Gaussian
    results_gauss = run_experiment(standard_gaussian, dim, n_samples, n_warmup, epsilon_L)
    plot_experiment_single(results_gauss, dim, 'Standard Gaussian', n_samples, 'hmc_step_size_comparison_gaussian.png')
    
    # Donut
    results_donut = run_experiment(lambda d: donut_distribution(d, radius=3.0, sigma2=0.5), dim, n_samples, n_warmup, epsilon_L)
    plot_experiment_single(results_donut, dim, 'Donut Distribution', n_samples, 'hmc_step_size_comparison_donut.png')

if __name__ == "__main__":
    main()

