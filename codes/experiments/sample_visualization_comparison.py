import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time
import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from algos.metropolis import MetropolisSampler, NormalProposal
from algos.hmc import TargetDistribution
from dists.banana import BananaDistribution
from dists.donut import DonutDistribution
from experiments.integrator_comparison import HMCWithIntegrator

class StudentTProposal:
    """Student-t proposal distribution."""
    def __init__(self, df: float = 3.0, scale: float = 1.0):
        self.df = df
        self.scale = scale
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x + self.scale * np.random.standard_t(df=self.df, size=x.shape)

class UniformProposal:
    """Uniform proposal distribution."""
    def __init__(self, width: float = 2.0):
        self.width = width
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x + np.random.uniform(-self.width, self.width, size=x.shape)

def compute_ess(samples: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Compute Effective Sample Size using autocorrelation."""
    if len(samples) <= 1:
        return 0.0
        
    n = len(samples)
    if max_lag is None:
        max_lag = min(50, n // 3)
        
    mean = np.mean(samples, axis=0)
    var = np.var(samples, axis=0)
    
    if np.all(var == 0):
        return 0.0
        
    normalized = (samples - mean) / np.sqrt(var)
    
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        c = np.mean(normalized[lag:] * normalized[:-lag if lag > 0 else None])
        if np.isnan(c):
            break
        acf[lag] = c
        
    # Find first negative or zero autocorrelation
    cutoff = np.argmin(acf > 0)
    if cutoff == 0:  # All positive
        cutoff = len(acf)
        
    # Sum up to cutoff
    rho = 1 + 2 * np.sum(acf[1:cutoff])
    
    return n / rho

def run_comparison(
    target: TargetDistribution,
    initial: np.ndarray,
    n_samples: int = 5000,
    metropolis_scales: List[float] = [0.1, 0.5, 1.0],
    hmc_step_sizes: List[float] = [0.1],
    hmc_n_steps: List[int] = [50],
    plot_title: str = "Sample Comparison"
) -> None:
    """Run and visualize comparison between different sampling methods."""
    
    dim = len(initial)
    if dim != 2:
        raise ValueError("This visualization only works for 2D distributions")
        
    methods = []
    samples_list = []
    labels = []
    stats = []
    
    # Metropolis with different proposals
    for scale in metropolis_scales:
        proposals = [
            ("Normal", NormalProposal(scale)),
            ("Student-t", StudentTProposal(scale=scale)),
            ("Uniform", UniformProposal(width=scale))
        ]
        
        for name, proposal in proposals:
            print(f"\nRunning Metropolis with {name} proposal (scale={scale})...")
            sampler = MetropolisSampler(
                target=target,
                proposal=proposal,
                initial=lambda: initial,
                iterations=n_samples
            )
            
            start_time = time.time()
            samples = sampler.run()
            end_time = time.time()
            
            samples = np.array(samples)
            ess = np.mean([compute_ess(samples[:, i]) for i in range(dim)])
            
            methods.append(f"Metropolis-{name}")
            samples_list.append(samples)
            labels.append(f"{name}\n(scale={scale})")
            stats.append({
                'method': f"Metropolis-{name}",
                'scale': scale,
                'acc_rate': sampler.acceptance_rate,
                'ess': ess,
                'time': end_time - start_time
            })
    
    # HMC with different integrators
    integrators = ['leapfrog', 'euler', 'modified_euler']
    
    for step_size in hmc_step_sizes:
        for n_steps in hmc_n_steps:
            for integrator in integrators:
                print(f"\nRunning HMC with {integrator} integrator (ε={step_size}, L={n_steps})...")
                sampler = HMCWithIntegrator(
                    target=target,
                    initial=lambda: initial,
                    integrator=integrator,
                    iterations=n_samples,
                    step_size=step_size,
                    L=n_steps
                )
                
                start_time = time.time()
                samples = sampler.run()
                end_time = time.time()
                
                samples = np.array(samples)
                ess = np.mean([compute_ess(samples[:, i]) for i in range(dim)])
                
                methods.append(f"HMC-{integrator}")
                samples_list.append(samples)
                labels.append(f"HMC-{integrator}\n(ε={step_size}, L={n_steps})")
                stats.append({
                    'method': f"HMC-{integrator}",
                    'step_size': step_size,
                    'n_steps': n_steps,
                    'acc_rate': sampler.get_acceptance_rate(),
                    'ess': ess,
                    'time': end_time - start_time
                })
    
    # Create visualization
    n_methods = len(methods)
    n_rows = (n_methods + 2) // 3  # Ensure at least 3 columns
    n_cols = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()
    
    # Plot target density contours
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = target(np.array([X[j, i], Y[j, i]]))
    
    for i, (method, samples, label) in enumerate(zip(methods, samples_list, labels)):
        ax = axes[i]
        # Plot target contours
        ax.contour(X, Y, Z, levels=10, colors='r', alpha=0.3)
        # Plot samples
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1)
        ax.set_title(label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Add statistics as text
        stat = stats[i]
        stat_text = f"Acc. Rate: {stat['acc_rate']:.2%}\nESS: {stat['ess']:.1f}\nESS/s: {stat['ess']/stat['time']:.1f}"
        ax.text(0.02, 0.98, stat_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove empty subplots
    for i in range(len(methods), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(plot_title)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{plot_title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare sampling methods with visualizations")
    parser.add_argument('--distribution', choices=['donut', 'banana'], default='donut',
                      help='Target distribution to sample from')
    parser.add_argument('--n_samples', type=int, default=5000,
                      help='Number of samples to generate')
    parser.add_argument('--metropolis_scales', type=float, nargs='+', default=[0.1, 0.5, 1.0],
                      help='Scales for Metropolis proposals')
    parser.add_argument('--hmc_step_sizes', type=float, nargs='+', default=[0.1],
                      help='Step sizes for HMC')
    parser.add_argument('--hmc_n_steps', type=int, nargs='+', default=[50],
                      help='Number of integration steps for HMC')
    
    args = parser.parse_args()
    
    # Set up target distribution
    if args.distribution == 'donut':
        target = DonutDistribution(radius=3.0, sigma2=0.05)  # Match donut_comparison parameters
        initial = np.array([3.0, 0.0])  # Start on x-axis at target radius
        title = "Donut Distribution Sampling Comparison"
    else:  # banana
        target = BananaDistribution()
        initial = np.array([0.0, 0.0])
        title = "Banana Distribution Sampling Comparison"
    
    run_comparison(
        target=target,
        initial=initial,
        n_samples=args.n_samples,
        metropolis_scales=args.metropolis_scales,
        hmc_step_sizes=args.hmc_step_sizes,
        hmc_n_steps=args.hmc_n_steps,
        plot_title=title
    )

if __name__ == "__main__":
    main() 