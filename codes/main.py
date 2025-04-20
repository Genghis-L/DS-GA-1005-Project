import numpy as np
import argparse
from typing import Callable, List, Any, Union
from algos.metropolis import MetropolisSampler, NormalProposal
from algos.hmc import HMCSampler
from dists import NormalDistribution, BananaDistribution, MixtureDistribution, BaseDistribution
from algos.visualization import plot_samples_1d, plot_samples_2d, plot_trace

def create_target_distribution(
    name: str,
    dim: int,
    **kwargs
) -> BaseDistribution:
    """Create a target distribution based on name and parameters."""
    if name == "normal":
        mean = np.zeros(dim)
        cov = np.eye(dim)
        if "mean" in kwargs:
            mean = np.array(kwargs["mean"])
        if "cov" in kwargs:
            cov = np.array(kwargs["cov"])
        return NormalDistribution(mean=mean, cov=cov)
    elif name == "banana":
        if dim != 2:
            raise ValueError("Banana distribution only supports 2D")
        return BananaDistribution(
            a=kwargs.get("a", 1.0),
            b=kwargs.get("b", 1.0)
        )
    elif name == "mixture":
        n_components = kwargs.get("n_components", 2)
        means = [np.random.normal(0, 3, dim) for _ in range(n_components)]
        covs = [np.eye(dim) for _ in range(n_components)]
        weights = np.ones(n_components) / n_components
        return MixtureDistribution(means, covs, weights)
    else:
        raise ValueError(f"Unknown distribution: {name}")

def run_mcmc(
    target: Union[Callable[[np.ndarray], float], BaseDistribution],
    initial: Callable[[], np.ndarray],
    sampler_type: str = "metropolis",
    iterations: int = 10_000,
    visualize: bool = True,
    save_plots: bool = False,
    **kwargs
) -> List[np.ndarray]:
    """
    Run MCMC sampling using either Metropolis or HMC.
    
    Args:
        target: Target distribution
        initial: Function that returns initial sample
        sampler_type: Either "metropolis" or "hmc"
        iterations: Number of iterations to run
        visualize: Whether to plot the results
        save_plots: Whether to save the plots
        **kwargs: Additional arguments for the specific sampler
        
    Returns:
        List of samples from the target distribution
    """
    # Prepare sampler-specific parameters
    sampler_kwargs = {"iterations": iterations}
    
    if sampler_type == "metropolis":
        # For Metropolis, we use step_size only for the proposal distribution
        proposal_scale = kwargs.get("scale_proposal", 0.1)
        sampler_kwargs["proposal"] = NormalProposal(scale=proposal_scale)
            
        sampler = MetropolisSampler(
            target=target,
            initial=initial,
            **sampler_kwargs
        )
    elif sampler_type == "hmc":
        # For HMC, we pass step_size and L directly
        sampler_kwargs.update({
            "step_size": kwargs.get("step_size", 0.1),
            "L": kwargs.get("L", 50)
        })
            
        sampler = HMCSampler(
            target=target,
            initial=initial,
            **sampler_kwargs
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
        
    samples = sampler.run()
    
    if visualize:
        dim = len(samples[0])
        prefix = f"{sampler_type}_{target.__class__.__name__}"
        
        # Plot trace
        plot_trace(
            samples,
            title=f"Trace Plot - {target.__class__.__name__}",
            save_path=f"{prefix}_trace.png" if save_plots else None
        )
        
        # Plot samples
        if dim == 1:
            plot_samples_1d(
                samples,
                target=target,
                title=f"Samples - {target.__class__.__name__}",
                save_path=f"{prefix}_samples.png" if save_plots else None
            )
        elif dim == 2:
            plot_samples_2d(
                samples,
                target=target,
                title=f"Samples - {target.__class__.__name__}",
                save_path=f"{prefix}_samples.png" if save_plots else None
            )
    
    return samples

def main():
    parser = argparse.ArgumentParser(description="Run MCMC sampling")
    parser.add_argument("--distribution", type=str, default="normal",
                       choices=["normal", "banana", "mixture"],
                       help="Target distribution to sample from")
    parser.add_argument("--dim", type=int, default=1,
                       help="Dimension of the probability space")
    parser.add_argument("--sampler", type=str, default="metropolis",
                       choices=["metropolis", "hmc"],
                       help="MCMC sampler to use")
    parser.add_argument("--iterations", type=int, default=10000,
                       help="Number of MCMC iterations")
    parser.add_argument("--step_size", type=float, default=0.1,
                       help="Step size for HMC or proposal scale for Metropolis")
    parser.add_argument("--scale_proposal", type=float, default=0.1,
                       help="Scale for the proposal distribution")
    parser.add_argument("--L", type=int, default=50,
                       help="Number of leapfrog steps for HMC")
    parser.add_argument("--visualize", action="store_true",
                       help="Whether to visualize the results")
    parser.add_argument("--save_plots", action="store_true",
                       help="Whether to save the plots")
    
    args = parser.parse_args()
    
    # Create target distribution
    target = create_target_distribution(
        args.distribution,
        args.dim,
        n_components=2  # For mixture distribution
    )
    
    # Create initial point
    initial = lambda: np.zeros(args.dim)
    
    # Run MCMC with sampler-specific parameters
    kwargs = {
        "step_size": args.step_size,
        "scale_proposal": args.scale_proposal
    }
    if args.sampler == "hmc":
        kwargs["L"] = args.L
    
    samples = run_mcmc(
        target=target,
        initial=initial,
        sampler_type=args.sampler,
        iterations=args.iterations,
        visualize=args.visualize,
        save_plots=args.save_plots,
        **kwargs
    )
    
    # Print summary statistics
    samples_array = np.array(samples)
    print("\nSummary Statistics:")
    print(f"Mean: {np.mean(samples_array, axis=0)}")
    print(f"Std: {np.std(samples_array, axis=0)}")
    print(f"Acceptance Rate: {len(np.unique(samples_array, axis=0)) / len(samples_array):.2%}")

if __name__ == "__main__":
    main() 