# Experiment Settings for HMC Integrator Comparison

## General Settings
| Parameter | Value |
|-----------|-------|
| Number of Samples | 1,000 (default) |
| Dimensions Tested | 2 (default, configurable) |
| Target Standard Deviation | 1.0 (for Gaussian) |
| Donut Radius | 3.0 (for Donut) |
| Donut Shell Thickness (sigma^2) | 0.05 (for Donut) |
| Step Sizes Tested | [0.1, 0.5, 1.0] (default, configurable) |
| Leapfrog Steps (L) | 20 (default, configurable) |

## Target Distributions

### Standard Gaussian
| Parameter | Value |
|-----------|-------|
| Type | Multivariate Normal |
| Mean | Zero vector |
| Covariance | Identity matrix scaled by target_std^2 |

### Donut Distribution
| Parameter | Value |
|-----------|-------|
| Type | N-dimensional shell |
| Radius | 3.0 |
| Shell Thickness (sigma^2) | 0.05 |

## HMC Integrator Settings
| Parameter | Value |
|-----------|-------|
| Integrators Compared | Euler, Modified Euler, Leapfrog |
| Leapfrog Steps (L) | 20 |
| Step Sizes | [0.1, 0.5, 1.0] (default, configurable) |
| Mass Matrix | Identity |
| Kinetic Energy | Gaussian |
| Initial State | Zero vector (Gaussian), [radius, 0, ..., 0] (Donut) |

## Performance Metrics
- Acceptance Rate
- Computation Time per Sample (μs)
- Effective Sample Size (ESS)
- Energy Conservation Error (relative energy error)

## Implementation Notes
- All distributions use NumPy's random number generators
- Each integrator is tested for each step size and target
- ESS is computed for each dimension and averaged
- Energy error is computed as the mean relative error over 100 test trajectories
- 2D sample plots and metric plots are generated for each integrator and step size 