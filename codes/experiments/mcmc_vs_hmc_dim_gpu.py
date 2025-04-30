import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add the parent directory to the path to import from algos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Target Distributions ---
class TargetDistribution:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self(x))
    def grad_log_density(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)
        logp = self.log_density(x)
        grad = torch.autograd.grad(logp, x)[0]
        return grad

class StandardGaussian(TargetDistribution):
    def __init__(self, dim: int):
        self.dim = dim
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * torch.sum(x * x, dim=-1)) / (2 * torch.pi) ** (self.dim / 2)
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(x * x, dim=-1) - (self.dim / 2) * torch.log(2 * torch.pi)
    def grad_log_density(self, x: torch.Tensor) -> torch.Tensor:
        return -x

class DonutDistribution(TargetDistribution):
    def __init__(self, dim: int, radius: float = 3.0, sigma2: float = 0.5):
        self.dim = dim
        self.radius = radius
        self.sigma2 = sigma2
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(torch.sum(x * x, dim=-1))
        return torch.exp(-0.5 * ((r - self.radius) ** 2) / self.sigma2)
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(torch.sum(x * x, dim=-1))
        return -0.5 * ((r - self.radius) ** 2) / self.sigma2
    def grad_log_density(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(torch.sum(x * x))
        if r == 0:
            return torch.zeros_like(x)
        return ((self.radius - r) / (self.sigma2 * r)) * x

# --- Samplers ---
class MetropolisSampler:
    def __init__(self, target, initial, proposal, iterations):
        self.target = target
        self.initial = initial
        self.proposal = proposal
        self.iterations = iterations
        self.samples = []
        self.accepted = 0
    def run(self):
        x = self.initial().to(device)
        self.samples = [x.clone()]
        for _ in range(self.iterations):
            x_new = self.proposal(x)
            x_new = x_new.to(device)
            log_alpha = self.target.log_density(x_new) - self.target.log_density(x)
            if torch.log(torch.rand(1, device=device)) < log_alpha:
                x = x_new
                self.accepted += 1
            self.samples.append(x.clone())
        return self.samples
    def get_acceptance_rate(self):
        return self.accepted / self.iterations

class HMCSampler:
    def __init__(self, target, initial, iterations, step_size, trajectory_length, kinetic_energy, kinetic_params=None):
        self.target = target
        self.initial = initial
        self.iterations = iterations
        self.step_size = step_size
        self.trajectory_length = trajectory_length
        self.L = int(trajectory_length // step_size) + 1
        self.kinetic_energy = kinetic_energy
        self.kinetic_params = kinetic_params or {}
        self.samples = []
        self.accepted = 0
    def _sample_momentum(self, dim):
        if self.kinetic_energy == 'gaussian':
            return torch.randn(dim, device=device)
        elif self.kinetic_energy == 'student_t':
            nu = self.kinetic_params.get('nu', 5.0)
            return torch.distributions.StudentT(nu).sample((dim,)).to(device)
        else:
            raise NotImplementedError
    def _kinetic_energy(self, r):
        if self.kinetic_energy == 'gaussian':
            return 0.5 * torch.sum(r * r)
        elif self.kinetic_energy == 'student_t':
            nu = self.kinetic_params.get('nu', 5.0)
            return (nu / 2) * torch.log1p(torch.sum(r * r) / nu)
        else:
            raise NotImplementedError
    def _kinetic_grad(self, r):
        if self.kinetic_energy == 'gaussian':
            return r
        elif self.kinetic_energy == 'student_t':
            nu = self.kinetic_params.get('nu', 5.0)
            return r / (1 + torch.sum(r * r) / nu)
        else:
            raise NotImplementedError
    def _leapfrog(self, theta, r):
        theta = theta.clone().detach().requires_grad_(True)
        r = r.clone().detach()
        r = r + (self.step_size / 2) * self.target.grad_log_density(theta)
        for _ in range(self.L):
            theta = theta + self.step_size * self._kinetic_grad(r)
            theta = theta.clone().detach().requires_grad_(True)
            if _ < self.L - 1:
                r = r + self.step_size * self.target.grad_log_density(theta)
        r = r + (self.step_size / 2) * self.target.grad_log_density(theta)
        return theta.detach(), r.detach()
    def _hamiltonian(self, theta, r):
        return -self.target.log_density(theta) + self._kinetic_energy(r)
    def run(self):
        x = self.initial().to(device)
        self.samples = [x.clone()]
        for _ in range(self.iterations):
            r0 = self._sample_momentum(x.shape[0])
            x0, r0 = x.clone(), r0.clone()
            x_new, r_new = self._leapfrog(x0, r0)
            h0 = self._hamiltonian(x0, r0)
            h_new = self._hamiltonian(x_new, r_new)
            log_alpha = h0 - h_new
            if torch.log(torch.rand(1, device=device)) < log_alpha:
                x = x_new
                self.accepted += 1
            self.samples.append(x.clone())
        return self.samples
    def get_acceptance_rate(self):
        return self.accepted / self.iterations

# --- Utility Functions ---
def compute_ess(samples: torch.Tensor) -> float:
    samples = samples.cpu().numpy()
    n = len(samples)
    if n <= 1:
        return 0.0
    max_lag = min(50, n // 3)
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

def run_experiment(target_class, dims: List[int], n_samples: int = 1000, n_warmup: int = 100) -> Dict[str, Dict[str, List[float]]]:
    results = {
        'mcmc': {
            'gaussian': {'acceptance': [], 'ess': [], 'time': []},
            'student_t': {'acceptance': [], 'ess': [], 'time': []},
            'uniform': {'acceptance': [], 'ess': [], 'time': []}
        },
        'hmc': {
            'gaussian': {'acceptance': [], 'ess': [], 'time': []},
            'student_t': {'acceptance': [], 'ess': [], 'time': []}
        }
    }
    for dim in dims:
        print(f"Running experiment for dimension {dim}")
        target = target_class(dim)
        # MCMC
        for proposal_type in ['gaussian', 'student_t', 'uniform']:
            if proposal_type == 'gaussian':
                proposal = lambda x: x + torch.randn(dim, device=device)
            elif proposal_type == 'student_t':
                proposal = lambda x: x + torch.distributions.StudentT(3).sample((dim,)).to(device)
            else:
                proposal = lambda x: x + (2 * torch.rand(dim, device=device) - 1)
            if isinstance(target, DonutDistribution):
                initial = lambda: torch.cat([torch.tensor([target.radius], device=device), torch.zeros(dim-1, device=device)])
            else:
                initial = lambda: torch.zeros(dim, device=device)
            start_time = time.time()
            sampler = MetropolisSampler(target, initial, proposal, n_samples + n_warmup)
            samples = torch.stack(sampler.run()[n_warmup:])
            end_time = time.time()
            results['mcmc'][proposal_type]['acceptance'].append(sampler.get_acceptance_rate())
            results['mcmc'][proposal_type]['ess'].append(compute_ess(samples))
            results['mcmc'][proposal_type]['time'].append(end_time - start_time)
        # HMC
        for kinetic_type in ['gaussian', 'student_t']:
            if kinetic_type == 'gaussian':
                kinetic_params = None
            else:
                kinetic_params = {'nu': 5.0}
            if isinstance(target, DonutDistribution):
                initial = lambda: torch.cat([torch.tensor([target.radius], device=device), torch.zeros(dim-1, device=device)])
            else:
                initial = lambda: torch.zeros(dim, device=device)
            start_time = time.time()
            sampler = HMCSampler(target, initial, n_samples + n_warmup, 0.1, 0.5, kinetic_type, kinetic_params)
            samples = torch.stack(sampler.run()[n_warmup:])
            end_time = time.time()
            results['hmc'][kinetic_type]['acceptance'].append(sampler.get_acceptance_rate())
            results['hmc'][kinetic_type]['ess'].append(compute_ess(samples))
            results['hmc'][kinetic_type]['time'].append(end_time - start_time)
    return results

def plot_results(results: Dict[str, Dict[str, List[float]]], dims: List[int], target_name: str, save_path: str, n_samples: int = 1000):
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f"Target Distribution: {target_name}\nNumber of Points to sample: {n_samples}", fontsize=16, y=0.98)
    mcmc_colors = ['#FF0000', '#FF8C00', '#FFD700']
    hmc_colors = ['#000080', '#0000FF']
    # Acceptance Rate
    ax = axes[0]
    for j, proposal in enumerate(results['mcmc'].keys()):
        label = f"MCMC - {proposal}"
        ax.plot(dims, results['mcmc'][proposal]['acceptance'], label=label, color=mcmc_colors[j], linestyle='--', marker='s', linewidth=2, markersize=6)
    for j, (kinetic, color) in enumerate(zip(results['hmc'].keys(), hmc_colors)):
        label = "HMC - Gaussian" if kinetic == 'gaussian' else "HMC - Student-t (ν=5)"
        ax.plot(dims, results['hmc'][kinetic]['acceptance'], label=label, color=color, linestyle='-', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Dimension (log scale)')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Acceptance Rate')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    # ESS
    ax = axes[1]
    for j, proposal in enumerate(results['mcmc'].keys()):
        label = f"MCMC - {proposal}"
        ax.plot(dims, results['mcmc'][proposal]['ess'], label=label, color=mcmc_colors[j], linestyle='--', marker='s', linewidth=2, markersize=6)
    for j, (kinetic, color) in enumerate(zip(results['hmc'].keys(), hmc_colors)):
        label = "HMC - Gaussian" if kinetic == 'gaussian' else "HMC - Student-t (ν=5)"
        ax.plot(dims, results['hmc'][kinetic]['ess'], label=label, color=color, linestyle='-', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Dimension (log scale)')
    ax.set_ylabel('ESS')
    ax.set_title('Effective Sample Size')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    # Time
    ax = axes[2]
    for j, proposal in enumerate(results['mcmc'].keys()):
        label = f"MCMC - {proposal}"
        ax.plot(dims, results['mcmc'][proposal]['time'], label=label, color=mcmc_colors[j], linestyle='--', marker='s', linewidth=2, markersize=6)
    for j, (kinetic, color) in enumerate(zip(results['hmc'].keys(), hmc_colors)):
        label = "HMC - Gaussian" if kinetic == 'gaussian' else "HMC - Student-t (ν=5)"
        ax.plot(dims, results['hmc'][kinetic]['time'], label=label, color=color, linestyle='-', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Dimension (log scale)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.9, hspace=0.4)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    dims = [2, 5, 10, 20]
    n_samples = 1000
    n_warmup = 100
    targets = {
        'Standard Gaussian': StandardGaussian,
        'Donut': DonutDistribution,
    }
    for target_name, target_class in targets.items():
        print(f"\nRunning experiments for {target_name}")
        results = run_experiment(target_class, dims, n_samples, n_warmup)
        plot_results(results, dims, target_name, f"{target_name.lower().replace(' ', '_')}_comparison_gpu.png", n_samples)

if __name__ == "__main__":
    main() 