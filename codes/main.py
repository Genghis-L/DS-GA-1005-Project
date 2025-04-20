import numpy as np
from typing import Callable, List, Any, Union
from algos.metropolis import MetropolisSampler, NormalProposal
from algos.hmc import HMCSampler
from dists import NormalDistribution, BananaDistribution, MixtureDistribution, BaseDistribution

def run_mcmc(
    target: Union[Callable[[np.ndarray], float], BaseDistribution],
    initial: Callable[[], np.ndarray],
    sampler_type: str = "metropolis",
    iterations: int = 10_000,
    **kwargs
) -> List[np.ndarray]:
    """
    Run MCMC sampling using either Metropolis or HMC.
    
    Args:
        target: Target distribution. For Metropolis, a callable that returns the density.
               For HMC, a BaseDistribution instance with grad_log_density implemented.
        initial: Function that returns initial sample
        sampler_type: Either "metropolis" or "hmc"
        iterations: Number of iterations to run
        **kwargs: Additional arguments for the specific sampler
        
    Returns:
        List of samples from the target distribution
    """
    if sampler_type == "metropolis":
        # Default proposal for Metropolis if not specified
        if "proposal" not in kwargs:
            kwargs["proposal"] = NormalProposal(scale=0.1)
            
        sampler = MetropolisSampler(
            target=target,
            initial=initial,
            iterations=iterations,
            **kwargs
        )
    elif sampler_type == "hmc":
        # Default HMC parameters if not specified
        if "L" not in kwargs:
            kwargs["L"] = 50
        if "step_size" not in kwargs:
            kwargs["step_size"] = 0.1
            
        sampler = HMCSampler(
            target=target,
            initial=initial,
            iterations=iterations,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
        
    return sampler.run()

# Example usage:
if __name__ == "__main__":
    # Example 1: Normal distribution with Metropolis
    normal = NormalDistribution(mean=np.array([0.0]), cov=np.array([[1.0]]))
    initial = lambda: np.array([0.0])
    samples = run_mcmc(
        target=normal,
        initial=initial,
        sampler_type="metropolis",
        iterations=1000
    )
    
    # Example 2: Normal distribution with HMC
    samples = run_mcmc(
        target=normal,
        initial=initial,
        sampler_type="hmc",
        iterations=1000
    )
    
    # Example 3: Banana distribution with HMC
    banana = BananaDistribution(a=1.0, b=1.0)
    initial = lambda: np.array([0.0, 0.0])
    samples = run_mcmc(
        target=banana,
        initial=initial,
        sampler_type="hmc",
        iterations=1000,
        step_size=0.1,
        L=20
    )
    
    # Example 4: Mixture of Gaussians with Metropolis
    means = [np.array([-2.0]), np.array([2.0])]
    covs = [np.array([[1.0]]), np.array([[1.0]])]
    mixture = MixtureDistribution(means, covs, weights=[0.3, 0.7])
    initial = lambda: np.array([0.0])
    samples = run_mcmc(
        target=mixture,
        initial=initial,
        sampler_type="metropolis",
        iterations=1000,
        proposal=NormalProposal(scale=0.5)
    ) 