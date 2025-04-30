# Experiment Settings for Metropolis Proposal Comparison

## General Settings
| Parameter | Value |
|-----------|-------|
| Number of Samples | 10,000 (default) |
| Dimensions Tested | 1 (default, configurable) |
| Target Standard Deviation | 1.0 (default, configurable) |
| Proposal Types | Normal, Student-t (df=3), Uniform |
| Proposal Scales | [0.1, 0.5, 1.0, 2.0, 5.0] (default, configurable) |

## Target Distribution

### Standard Gaussian
| Parameter | Value |
|-----------|-------|
| Type | Multivariate Normal |
| Mean | Zero vector |
| Covariance | Identity matrix scaled by target_std^2 |

## Metropolis Algorithm Settings
| Parameter | Value |
|-----------|-------|
| Initial State | Zero vector |
| Iterations | Number of samples specified (default: 10,000) |
| Proposal Distributions | Normal, Student-t (df=3), Uniform |
| Proposal Scale | As specified in the experiment (see above) |
| Acceptance Rate | Computed and reported per run |
| ESS Calculation | Autocorrelation up to lag min(50, n//3) |
| Warmup | None (all samples used) |

## Performance Metrics
- Acceptance Rate
- Time per Sample (μs)
- Effective Sample Size (ESS)
- ESS per Second

## Implementation Notes
- All distributions use NumPy's random number generators
- ESS is computed for each dimension and averaged
- Trajectory plots are generated for the first coordinate
- Results are grouped and plotted by proposal type and scale 