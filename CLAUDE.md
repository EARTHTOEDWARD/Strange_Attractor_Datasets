# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a comprehensive toolkit for generating chaotic datasets from dynamical systems, specifically designed for machine learning research and chaos theory applications. The repository implements three major dynamical systems: Lorenz (atmospheric), Rössler (chemical), and Hindmarsh-Rose (neuronal).

## Core Architecture

### Dataset Generation Scripts
The repository contains five main executable scripts, each serving different purposes:

1. **`strange_attractor_generator.py`** - The comprehensive toolkit with class-based architecture:
   - `LorenzGenerator`: Handles Lorenz-63 system integration and parameter sweeps
   - `TimeDelayEmbedding`: Converts 1D signals to phase space coordinates
   - `DatasetManager`: Manages dataset saving with automatic metadata generation

2. **`chaotic_dataset_example.py`** - Self-contained demonstration script for basic Lorenz trajectories and time-delay embedding

3. **`rossler_advanced_dataset.py`** - Advanced Rössler system generator with Lyapunov exponent calculation

4. **`hindmarsh_rose_dataset.py`** - Neuronal dynamics generator with chaos analysis across current regimes

5. **`embed_eeg_chbmit.py`** - Real-world EEG data processor for CHB-MIT epilepsy dataset:
   - Reads .edf files using MNE-Python
   - Applies time-delay embedding to individual EEG channels
   - Segments data by seizure/baseline periods using CSV annotations
   - Outputs labeled phase space coordinates for machine learning

### Data Architecture
All generated datasets follow consistent patterns:
- **CSV format** with standardized column naming (`t`, `x`, `y`, `z` for trajectories)
- **Parameter sweeps** create separate files per parameter value with summary tables
- **Lyapunov summaries** map parameter values to chaos measures
- **Automatic metadata** generation in JSON format for reproducibility

## Common Development Commands

### Generate Complete Datasets
```bash
# Generate all three dynamical systems datasets
python strange_attractor_generator.py     # Lorenz + embedding examples
python rossler_advanced_dataset.py        # Rössler with Lyapunov analysis  
python hindmarsh_rose_dataset.py          # Neuronal dynamics dataset

# Quick demonstration (self-contained)
python chaotic_dataset_example.py         # Basic Lorenz + embedding

# Real-world EEG data processing (requires MNE-Python)
python embed_eeg_chbmit.py --edf file.edf --channel F7 --tau 30 --dim 3

# Test EEG functionality without real data
python test_embed_eeg.py                  # Synthetic EEG embedding demo
```

### Dependencies Installation
```bash
# Core dependencies for synthetic datasets
pip install numpy scipy pandas matplotlib

# Additional dependency for real EEG processing
pip install mne
```

### Dataset Verification
```bash
# Check generated file counts
find data -name "*.csv" | wc -l

# View dataset structure  
head -5 data/lorenz_parameter_sweep.csv
head -5 data/rossler_lyapunov_summary.csv
head -5 data/hr_lyapunov_summary.csv
```

## Key Implementation Details

### Numerical Integration
- **Lorenz system**: Uses `scipy.integrate.solve_ivp` with RK45 method and `rtol=1e-8`
- **Rössler/Hindmarsh-Rose**: Custom 4th-order Runge-Kutta implementation for precise control
- **Burn-in periods**: All systems include transient removal (1000-20000 steps depending on system)

### Lyapunov Exponent Calculation
Both Rössler and Hindmarsh-Rose implementations use the two-trajectory renormalization method:
- Parallel integration of perturbed trajectories
- Periodic renormalization (every 10-20 steps)
- Logarithmic divergence accumulation over long time series (6000-8000 time units)

### Parameter Sweep Strategy
- **Lorenz**: Focus on ρ parameter (10-40 range) to capture stable→chaotic transition at ρ≈24.7
- **Rössler**: Sweep c parameter (4-8 range) with chaos around c≈5.7  
- **Hindmarsh-Rose**: External current I_ext (1.0-3.5) capturing quiescent→spiking→chaotic regimes

## Dataset Scale and Structure

### Generated Data Volume
- **Lorenz**: ~170,000 trajectory points across parameter sweep
- **Rössler**: ~1.7 million points (17 parameters × 100k points each)
- **Hindmarsh-Rose**: ~2.6 million points (26 parameters × 100k points each)
- **EEG embeddings**: Variable size depending on input data (typically 100k-1M points per channel/segment)
- **Total repository size**: ~4.5 million labeled trajectory points (synthetic) + real-world EEG embeddings

### File Naming Conventions
- Trajectory files: `{system}_{parameter}{value}.csv` (e.g., `rossler_c5.75.csv`, `hr_I2.20.csv`)
- Summary files: `{system}_lyapunov_summary.csv`
- Visualization files: `{system}_{parameter}_{plot_type}.png`
- Metadata files: `{filename}_metadata.json`
- EEG embedding files: `{edf_name}_{channel}_{label}_{segment}.csv` where label = "seiz" or "base"

## Working with the Codebase

### Adding New Dynamical Systems
Extend the architecture by following the established patterns:
1. Implement equations as `{system}_rhs()` function
2. Add 4th-order RK integrator with burn-in
3. Include parameter sweep functionality
4. Add Lyapunov exponent calculation if needed
5. Follow consistent file naming and metadata generation

### Modifying Integration Parameters
Key parameters to adjust for different research needs:
- **Time spans**: Typically 5000-6000 time units for sufficient attractor sampling
- **Step sizes**: `dt=0.01-0.05` balances accuracy vs computation time
- **Sampling rates**: 100,000 points standard for ML-ready datasets
- **Burn-in periods**: System-dependent (1000 for Lorenz, 10000+ for others)

### Visualization Integration
All scripts include optional matplotlib integration with graceful fallback:
- Set `HAS_PLOT = True/False` based on matplotlib availability
- Generate publication-ready plots at 150 DPI
- Save visualizations alongside datasets for immediate analysis