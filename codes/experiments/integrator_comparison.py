import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.hmc import  HMCWithIntegrator, compute_relative_energy_error
from dists.normal import NormalDistribution
from dists.donut import DonutDistribution

def run_comparison(
    n_samples: int = 1000,
    dim: int = 2,
    target_dist: str = 'donut',
    target_std: float = 1.0,
    donut_radius: float = 3.0,
    donut_sigma2: float = 0.5,
    step_sizes: list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    trajectory_length: float = 0.5,
    output_file: str = None
):
    """Run comparison of different HMC integrators."""
    # Setup target distribution
    if target_dist == 'normal':
        target = NormalDistribution(
            mean=np.zeros(dim),
            cov=target_std**2 * np.eye(dim)
        )
        initial = lambda: np.zeros(dim)
        plot_range = (-3*target_std, 3*target_std)
    else:  # donut
        target = DonutDistribution(
            radius=donut_radius,
            sigma2=donut_sigma2,
            dim=dim
        )
        initial = lambda: np.array([donut_radius] + [0.0] * (dim-1))  # Start on x-axis at radius
        plot_range = (-4, 4)  # Fixed range for donut visualization
    
    integrators = ['euler', 'modified_euler', 'leapfrog']
    results = []
    
    # Create figure for metrics comparison
    fig_metrics, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    fig_metrics.suptitle(f"HMC Integrator Comparison - {target_dist.title()} Distribution ({dim}D)")
    
    # Run comparison for each integrator and step size
    for step_idx, step_size in enumerate(step_sizes):
        for int_idx, integrator in enumerate(integrators):
            # Run sampler
            sampler = HMCWithIntegrator(
                target=target,
                initial=initial,
                integrator=integrator,
                iterations=n_samples,
                step_size=step_size,
                trajectory_length=trajectory_length
            )
            samples = np.array(sampler.run())
            
            # Compute metrics
            energy_error = compute_relative_energy_error(sampler)
            
            results.append({
                'integrator': integrator,
                'step_size': step_size,
                'acc_rate': sampler.get_acceptance_rate(),
                'energy_error': energy_error
            })
            
            print(f"\nResults for {integrator.upper()} (step_size={step_size}, trajectory_length={trajectory_length}):")
            print(f"  Acceptance rate: {sampler.get_acceptance_rate():.2%}")
            print(f"  Energy error: {energy_error:.2e}")
            print(f"  Number of leapfrog steps: {sampler.L}")
    
    # Group results by integrator for metric plots
    for integrator in integrators:
        int_results = [r for r in results if r['integrator'] == integrator]
        step_sizes_plot = [r['step_size'] for r in int_results]
        acc_rates = [r['acc_rate'] for r in int_results]
        errors = [r['energy_error'] for r in int_results]
        
        # Plot acceptance rate
        ax1.plot(step_sizes_plot, acc_rates, 'o-', label=integrator)
        ax1.set_xlabel('Step Size (ε)')
        ax1.set_ylabel('Acceptance Rate')
        ax1.set_title('Acceptance Rate vs Step Size')
        ax1.grid(True)
        ax1.legend()
        
        # Plot energy error
        ax2.plot(step_sizes_plot, errors, 'o-', label=integrator)
        ax2.set_xlabel('Step Size (ε)')
        ax2.set_ylabel('Relative Energy Error')
        ax2.set_title('Energy Conservation Error')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
    
    plt.tight_layout()
    
    # Save plots
    if output_file:
        fig_metrics.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Compare different HMC integrators")
    parser.add_argument("--distribution", choices=['normal', 'donut'], default='donut',
                       help="Target distribution (default: normal)")
    parser.add_argument("--dim", type=int, default=2,
                       help="Number of dimensions (default: 2)")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples to generate (default: 1000)")
    parser.add_argument("--target-std", type=float, default=1.0,
                       help="Standard deviation of target normal distribution (default: 1.0)")
    parser.add_argument("--donut-radius", type=float, default=3.0,
                       help="Target radius for donut distribution (default: 3.0)")
    parser.add_argument("--donut-sigma2", type=float, default=0.5,
                       help="Shell thickness for donut distribution (default: 0.5)")
    parser.add_argument("--step-sizes", type=float, nargs='+',
                       default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                       help="Step sizes (ε) to test")
    parser.add_argument("--trajectory-length", type=float, default=0.5,
                       help="Total trajectory length (s = ε * L) (default: 0.5)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file name (default: integrator_comparison_{dist}_{dim}d.png)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'integrator_comparison_{args.distribution}_{args.dim}d.png'
    
    print(f"\nRunning comparison for {args.distribution} distribution in {args.dim} dimensions...")
    print(f"Parameters:")
    if args.distribution == 'normal':
        print(f"  - Target std: {args.target_std}")
    else:
        print(f"  - Donut radius: {args.donut_radius}")
        print(f"  - Shell thickness: {args.donut_sigma2}")
    print(f"  - Number of samples: {args.n_samples}")
    print(f"  - Step sizes (ε): {args.step_sizes}")
    print(f"  - Trajectory length (s): {args.trajectory_length}")
    print(f"  - Output file: {args.output}\n")
    
    run_comparison(
        n_samples=args.n_samples,
        dim=args.dim,
        target_dist=args.distribution,
        target_std=args.target_std,
        donut_radius=args.donut_radius,
        donut_sigma2=args.donut_sigma2,
        step_sizes=args.step_sizes,
        trajectory_length=args.trajectory_length,
        output_file=args.output
    )

if __name__ == "__main__":
    main() 