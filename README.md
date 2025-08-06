# Strange Attractor Datasets

A comprehensive toolkit for generating chaotic datasets from dynamical systems and embedding real-world signals into phase space representations.

Based on "The Idiot's Guide to Building Your Own Strange-Attractor Datasets" methodology.

## Features

- **Lorenz System Generation**: Create chaotic trajectories from the famous Lorenz-63 equations
- **Parameter Sweeps**: Generate datasets across parameter ranges for classification/regression tasks
- **Time-Delay Embedding**: Convert 1D signals into phase space representations
- **Metadata Management**: Automatic documentation of dataset parameters and characteristics
- **Ready-to-Use**: CSV outputs compatible with any ML framework

## Quick Start

### Install Dependencies
```bash
pip install numpy scipy pandas matplotlib
```

### Generate Sample Datasets
```bash
python strange_attractor_generator.py
```

This creates three datasets in the `data/` folder:
- `lorenz_single_run.csv` - Single chaotic trajectory (10,000 points)
- `lorenz_parameter_sweep.csv` - Multiple trajectories with varying parameters (160,000 points)
- `sample_signal_embedded.csv` - Embedded synthetic signal (phase space coordinates)

## Usage

### 1. Lorenz System Datasets

```python
from strange_attractor_generator import LorenzGenerator

# Create generator
lorenz = LorenzGenerator(sigma=10.0, rho=28.0, beta=8.0/3.0)

# Single trajectory
data = lorenz.generate_single_trajectory(
    initial_state=[1.0, 1.0, 1.0],
    t_span=(0, 40),
    n_points=10000
)

# Parameter sweep for classification/regression
rho_values = np.linspace(10, 40, 16)
sweep_data = lorenz.generate_parameter_sweep('rho', rho_values)
```

### 2. Time-Delay Embedding

```python
from strange_attractor_generator import TimeDelayEmbedding

embedding = TimeDelayEmbedding()

# Embed any 1D signal
embedded_coords = embedding.embed_signal(
    signal=your_signal,
    tau=20,              # time delay in samples
    embedding_dim=3      # phase space dimension
)
```

### 3. Dataset Management

```python
from strange_attractor_generator import DatasetManager

manager = DatasetManager(base_path="my_datasets")

# Save with automatic metadata
manager.save_dataset(
    data=your_dataframe,
    filename="experiment_01",
    metadata={
        "description": "My experimental setup",
        "parameters": {"param1": value1, "param2": value2}
    }
)
```

## File Structure

```
Strange_Attractor_Datasets/
├── strange_attractor_generator.py    # Main toolkit
├── notebooks/
│   └── demo_notebook.ipynb          # Interactive examples
├── data/                            # Generated datasets
│   ├── *.csv                        # Dataset files
│   ├── *_metadata.json              # Individual metadata
│   └── dataset_info.json            # Combined metadata
└── README.md                        # This file
```

## Dataset Descriptions

### Lorenz Single Run
- **File**: `lorenz_single_run.csv`
- **Columns**: `t`, `x`, `y`, `z`
- **Points**: 10,000
- **Use Case**: Basic chaotic time series analysis, visualization

### Lorenz Parameter Sweep  
- **File**: `lorenz_parameter_sweep.csv`
- **Columns**: `t`, `x`, `y`, `z`, `rho`
- **Points**: 160,000 (16 parameter values × 10,000 points each)
- **Use Case**: Classification (stable vs chaotic), regression on parameter values

### Embedded Signal
- **File**: `sample_signal_embedded.csv`  
- **Columns**: `x1`, `x2`, `x3` (phase space coordinates)
- **Source**: Noisy sine wave (configurable)
- **Use Case**: Testing embedding techniques, signal processing

## Mathematical Background

### Lorenz-63 System
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y  
dz/dt = xy - βz
```
- Default parameters: σ=10, ρ=28, β=8/3
- Chaotic for ρ > 24.74 (approximately)

### Time-Delay Embedding
Converts signal `s(t)` to coordinates:
```
[s(t), s(t+τ), s(t+2τ), ..., s(t+(m-1)τ)]
```
- `τ`: time delay
- `m`: embedding dimension
- Reveals hidden attractor structure in 1D signals

## Machine Learning Applications

### Classification Tasks
- **Stable vs Chaotic**: Use `rho` parameter as threshold (ρ > 24.7 ≈ chaotic)
- **Parameter Regimes**: Classify by dynamical behavior type
- **Signal Types**: Distinguish embedded signals from different sources

### Regression Tasks  
- **Parameter Estimation**: Predict `rho`, `sigma`, `beta` from trajectory data
- **Lyapunov Exponents**: Predict chaos measures from short sequences
- **Noise Levels**: Estimate embedding quality metrics

### Time Series Tasks
- **Forecasting**: Predict next points in chaotic sequences
- **Anomaly Detection**: Identify deviations from expected attractor behavior
- **Dimensionality**: Estimate intrinsic dimension of embedded signals

## Advanced Usage

### Custom Dynamical Systems
Extend `LorenzGenerator` for other systems (Rössler, Mackey-Glass, etc.):

```python
def rossler_equations(self, t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return [dx, dy, dz]
```

### Real-World Data
Replace synthetic signals with actual data:

```python
# Load your signal
signal = np.loadtxt('my_eeg_data.csv')  # or audio, sensor data, etc.

# Embed and save
embedded = embedding.embed_signal(signal, tau=optimal_tau, embedding_dim=optimal_dim)
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Pandas ≥ 1.3
- Matplotlib ≥ 3.4

## Contributing

This toolkit is designed to be extended. Common additions:
- New dynamical systems
- Advanced embedding techniques (mutual information, false nearest neighbors)
- Chaos measures (Lyapunov exponents, correlation dimension)
- Visualization utilities
- ML preprocessing pipelines

## References

- Lorenz, E.N. (1963). "Deterministic Nonperiodic Flow"
- Takens, F. (1981). "Detecting Strange Attractors in Turbulence"
- Kantz, H. & Schreiber, T. (2004). "Nonlinear Time Series Analysis"

---

**Author**: Edward Farrelly  
**License**: MIT  
**Keywords**: chaos theory, strange attractors, time series, dynamical systems, machine learning