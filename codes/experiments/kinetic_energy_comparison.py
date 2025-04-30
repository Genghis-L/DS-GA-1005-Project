import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.hmc import HMCWithIntegrator, compute_relative_energy_error
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
    """Run comparison of different kinetic energy functions."""
    # Setup target distribution
    if target_dist == 'normal':
        target = NormalDistribution(
            mean=np.zeros(dim),
            cov=target_std**2 * np.eye(dim)
        )
        initial = lambda: np.zeros(dim)
    else:  # donut
        target = DonutDistribution(
            radius=donut_radius,
            sigma2=donut_sigma2,
            dim=dim
        )
        initial = lambda: np.array([donut_radius] + [0.0] * (dim-1))
    
    # Define kinetic energy configurations
    kinetic_configs = [
        ('gaussian', None),
        ('student_t', {'nu': 5}),  # df = 5 for Student's t
        ('alpha_norm', {'alpha': 1}),  # L1 norm
        ('alpha_norm', {'alpha': 3})   # L3 norm
    ]
    
    results = []
    
    # Create figure for metrics comparison
    fig_metrics, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig_metrics.suptitle(f"HMC Kinetic Energy Comparison - {target_dist.title()} Distribution ({dim}D)")
    
    # Run comparison for each kinetic energy and step size
    for step_idx, step_size in enumerate(step_sizes):
        for kinetic_type, params in kinetic_configs:
            # Create label for the kinetic energy type
            if kinetic_type == 'gaussian':
                label = 'Gaussian'
            elif kinetic_type == 'student_t':
                label = f'Student-t (ν={params["nu"]})'
            else:  # alpha_norm
                label = f'α-norm (α={params["alpha"]})'
            
            # Run sampler
            sampler = HMCWithIntegrator(
                target=target,
                initial=initial,
                integrator='leapfrog',  # Use leapfrog for all comparisons
                iterations=n_samples,
                step_size=step_size,
                trajectory_length=trajectory_length,
                kinetic_energy=kinetic_type,
                kinetic_params=params
            )
            samples = np.array(sampler.run())
            
            # Compute metrics
            energy_error = compute_relative_energy_error(sampler)
            
            results.append({
                'kinetic_type': label,
                'step_size': step_size,
                'acc_rate': sampler.get_acceptance_rate(),
                'energy_error': energy_error
            })
            
            print(f"\nResults for {label} (step_size={step_size}, trajectory_length={trajectory_length}):")
            print(f"  Acceptance rate: {sampler.get_acceptance_rate():.2%}")
            print(f"  Energy error: {energy_error:.2e}")
            print(f"  Number of leapfrog steps: {sampler.L}")
    
    # Group results by kinetic energy type for metric plots
    unique_types = sorted(set(r['kinetic_type'] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    
    for idx, kinetic_type in enumerate(unique_types):
        type_results = [r for r in results if r['kinetic_type'] == kinetic_type]
        step_sizes_plot = [r['step_size'] for r in type_results]
        acc_rates = [r['acc_rate'] for r in type_results]
        errors = [r['energy_error'] for r in type_results]
        
        # Plot acceptance rate
        ax1.plot(step_sizes_plot, acc_rates, 'o-', label=kinetic_type, color=colors[idx])
        ax1.set_xlabel('Step Size (ε)')
        ax1.set_ylabel('Acceptance Rate')
        ax1.set_title('Acceptance Rate vs Step Size')
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot energy error
        ax2.plot(step_sizes_plot, errors, 'o-', label=kinetic_type, color=colors[idx])
        ax2.set_xlabel('Step Size (ε)')
        ax2.set_ylabel('Relative Energy Error')
        ax2.set_title('Energy Conservation Error')
        ax2.set_yscale('log')  # Only y-axis in log scale
        ax2.grid(True)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plots
    if output_file:
        fig_metrics.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Compare different HMC kinetic energy functions")
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
                       help="Output file name (default: kinetic_energy_comparison_{dist}_{dim}d.png)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'kinetic_energy_comparison_{args.distribution}_{args.dim}d.png'
    
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