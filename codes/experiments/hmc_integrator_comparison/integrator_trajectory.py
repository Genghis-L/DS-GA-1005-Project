import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.hmc import TargetDistribution
from integrator_comparison import HMCWithIntegrator

class SimpleHarmonicOscillator(TargetDistribution):
    """Simple harmonic oscillator with H(q,p) = q²/2 + p²/2."""
    
    def __init__(self):
        pass
        
    def log_density(self, q: np.ndarray) -> float:
        """Log density is proportional to -q²/2."""
        return -0.5 * np.sum(q**2)
        
    def grad_log_density(self, q: np.ndarray) -> np.ndarray:
        """Gradient of log density is -q."""
        return -q

class HamiltonianTrajectory(HMCWithIntegrator):
    """Class to compute and visualize Hamiltonian trajectories."""
    
    def __init__(
        self,
        step_size: float = 0.3,
        n_steps: int = 20,
        integrator: str = 'leapfrog'
    ):
        target = SimpleHarmonicOscillator()
        initial = lambda: np.array([0.0])  # Will be overridden in plot_trajectories
        super().__init__(target, initial, integrator, iterations=1, L=n_steps, step_size=step_size)
    
    def compute_trajectory(self, q0: np.ndarray, p0: np.ndarray) -> tuple:
        """Compute trajectory using current integrator."""
        q = [q0]
        p = [p0]
        
        q_current, p_current = q0, p0
        for _ in range(self.L):
            # Use parent class's integrator methods for a single step
            q_next, p_next = self._integrate(q_current, p_current)
            
            q.append(q_next)
            p.append(p_next)
            q_current, p_current = q_next, p_next
            
        return np.array(q), np.array(p)
    
    def true_trajectory(self, q0: np.ndarray, p0: np.ndarray, n_points: int = 100) -> tuple:
        """Compute the true trajectory (circle in phase space)."""
        # For harmonic oscillator, trajectory is a circle
        radius = np.sqrt(self._compute_hamiltonian(q0, p0) * 2)
        angles = np.linspace(0, 2*np.pi, n_points)
        q = radius * np.cos(angles)
        p = radius * np.sin(angles)
        return q, p
    
    def plot_trajectories(
        self,
        q0: float = 0.0,
        p0: float = 1.0,
        output_file: str = None
    ):
        """Plot trajectories for different integrators."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hamiltonian Dynamics Integration Methods')
        
        # Initial conditions
        q0 = np.array([q0])
        p0 = np.array([p0])
        
        # True trajectory
        q_true, p_true = self.true_trajectory(q0, p0)
        
        # Plot settings
        methods = {
            'euler': ('Euler\'s method', (0, 0)),
            'modified_euler': ('Modified Euler\'s method', (0, 1)),
            'leapfrog_base': ('Leapfrog method', (0, 2)),
            'leapfrog_medium': ('Leapfrog method', (1, 0)),
            'leapfrog_large': ('Leapfrog method', (1, 1)),
            'leapfrog_xlarge': ('Leapfrog method', (1, 2))
        }
        
        # Define step sizes for each method
        step_sizes = {
            'euler': self.step_size,
            'modified_euler': self.step_size,
            'leapfrog_base': self.step_size,      # 0.3
            'leapfrog_medium': self.step_size * 2,  # 0.6
            'leapfrog_large': self.step_size * 4,   # 1.2
            'leapfrog_xlarge': self.step_size * 6  # 1.8
        }
        
        for method, (title, pos) in methods.items():
            ax = axes[pos[0], pos[1]]
            
            # Set method-specific step size and integrator
            orig_step_size = self.step_size
            self.step_size = step_sizes[method]
            self.integrator = 'leapfrog' if 'leapfrog' in method else method
            
            # Compute and plot trajectory
            q, p = self.compute_trajectory(q0, p0)
            
            # Plot
            ax.plot(q_true, p_true, 'gray', alpha=0.5, label='True trajectory')
            ax.plot(q, p, 'k.-', label='Numerical solution')
            ax.set_xlabel('Position (q)')
            ax.set_ylabel('Momentum (p)')
            
            # Set title based on method
            if 'leapfrog' in method:
                ax.set_title(f"Leapfrog method, stepsize {self.step_size:.1f}")
            else:
                ax.set_title(f"{title}, stepsize {self.step_size:.1f}")
            
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.grid(True)
            ax.axis('equal')
            
            # Reset step size
            self.step_size = orig_step_size
        
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Hamiltonian dynamics trajectories")
    parser.add_argument("--step-size", type=float, default=0.3,
                       help="Base step size for integration (default: 0.3)")
    parser.add_argument("--n-steps", type=int, default=20,
                       help="Number of integration steps (default: 20)")
    parser.add_argument("--q0", type=float, default=0.0,
                       help="Initial position (default: 0.0)")
    parser.add_argument("--p0", type=float, default=1.0,
                       help="Initial momentum (default: 1.0)")
    parser.add_argument("--output", type=str, default="integrator_trajectories.png",
                       help="Output file name (default: integrator_trajectories.png)")
    
    args = parser.parse_args()
    
    simulator = HamiltonianTrajectory(
        step_size=args.step_size,
        n_steps=args.n_steps
    )
    
    simulator.plot_trajectories(
        q0=args.q0,
        p0=args.p0,
        output_file=args.output
    )

if __name__ == "__main__":
    main() 