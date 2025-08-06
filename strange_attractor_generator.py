#!/usr/bin/env python3
"""
Strange Attractor Dataset Generator

A comprehensive toolkit for generating chaotic datasets from dynamical systems
and embedding real-world signals into phase space representations.

Author: Edward Farrelly
Based on "The Idiot's Guide to Building Your Own Strange-Attractor Datasets"
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
import json
from pathlib import Path


class LorenzGenerator:
    """Generate datasets from the Lorenz-63 dynamical system."""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        """
        Initialize Lorenz system parameters.
        
        Args:
            sigma: Prandtl number (default 10.0)
            rho: Rayleigh number (default 28.0)  
            beta: Physical parameter (default 8/3)
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def lorenz_equations(self, t: float, state: List[float], 
                        sigma: float = None, rho: float = None, beta: float = None) -> List[float]:
        """
        Lorenz-63 system equations.
        
        Args:
            t: Time (not used but required by scipy)
            state: Current state [x, y, z]
            sigma, rho, beta: Override default parameters if provided
            
        Returns:
            Derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        
        # Use provided parameters or defaults
        s = sigma if sigma is not None else self.sigma
        r = rho if rho is not None else self.rho
        b = beta if beta is not None else self.beta
        
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        
        return [dx, dy, dz]
    
    def generate_single_trajectory(self, 
                                 initial_state: List[float] = [1.0, 1.0, 1.0],
                                 t_span: Tuple[float, float] = (0, 40),
                                 n_points: int = 10000,
                                 **params) -> pd.DataFrame:
        """
        Generate a single Lorenz trajectory.
        
        Args:
            initial_state: Starting point [x0, y0, z0]
            t_span: Integration time span (start, end)
            n_points: Number of sample points
            **params: Override system parameters (sigma, rho, beta)
            
        Returns:
            DataFrame with columns ['t', 'x', 'y', 'z']
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Create args tuple with updated parameters
        current_params = [
            params.get('sigma', self.sigma),
            params.get('rho', self.rho), 
            params.get('beta', self.beta)
        ]
        
        sol = solve_ivp(
            lambda t, y: self.lorenz_equations(t, y, *current_params),
            t_span, 
            initial_state, 
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8
        )
        
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        
        return pd.DataFrame({
            't': sol.t,
            'x': sol.y[0],
            'y': sol.y[1], 
            'z': sol.y[2]
        })
    
    def generate_parameter_sweep(self,
                               param_name: str,
                               param_values: np.ndarray,
                               initial_state: List[float] = [1.0, 1.0, 1.0],
                               t_span: Tuple[float, float] = (0, 40),
                               n_points: int = 10000) -> pd.DataFrame:
        """
        Generate multiple trajectories by sweeping a parameter.
        
        Args:
            param_name: Parameter to sweep ('sigma', 'rho', or 'beta')
            param_values: Array of parameter values to try
            initial_state: Starting point for each trajectory
            t_span: Integration time span
            n_points: Points per trajectory
            
        Returns:
            DataFrame with additional column for the swept parameter
        """
        if param_name not in ['sigma', 'rho', 'beta']:
            raise ValueError("param_name must be 'sigma', 'rho', or 'beta'")
        
        trajectories = []
        
        for value in param_values:
            params = {param_name: value}
            
            df = self.generate_single_trajectory(
                initial_state=initial_state,
                t_span=t_span, 
                n_points=n_points,
                **params
            )
            
            # Add parameter column
            df[param_name] = value
            trajectories.append(df)
        
        return pd.concat(trajectories, ignore_index=True)


class TimeDelayEmbedding:
    """Convert 1D time series into phase space representations."""
    
    @staticmethod
    def embed_signal(signal: np.ndarray, 
                    tau: int, 
                    embedding_dim: int) -> np.ndarray:
        """
        Perform time-delay embedding on a 1D signal.
        
        Args:
            signal: 1D time series data
            tau: Time delay (in samples)
            embedding_dim: Embedding dimension
            
        Returns:
            Embedded coordinates as (N, embedding_dim) array
        """
        N = len(signal) - (embedding_dim - 1) * tau
        
        if N <= 0:
            raise ValueError("Signal too short for given tau and embedding_dim")
        
        embedded = np.zeros((N, embedding_dim))
        
        for i in range(embedding_dim):
            start_idx = i * tau
            end_idx = start_idx + N
            embedded[:, i] = signal[start_idx:end_idx]
        
        return embedded
    
    @staticmethod
    def create_sample_signal(duration: float = 10.0,
                           sampling_rate: float = 1000.0,
                           frequency: float = 5.0,
                           noise_level: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a sample noisy sine wave for testing.
        
        Args:
            duration: Signal duration in seconds
            sampling_rate: Samples per second
            frequency: Sine wave frequency in Hz
            noise_level: Noise amplitude relative to signal
            
        Returns:
            Tuple of (time_array, signal_array)
        """
        t = np.linspace(0, duration, int(sampling_rate * duration))
        signal = np.sin(2 * np.pi * frequency * t) + noise_level * np.random.randn(len(t))
        return t, signal


class DatasetManager:
    """Manage dataset creation, saving, and metadata."""
    
    def __init__(self, base_path: str = "data"):
        """Initialize with base directory for datasets."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.metadata = {}
    
    def save_dataset(self, 
                    data: pd.DataFrame, 
                    filename: str,
                    metadata: Dict[str, Any] = None) -> None:
        """
        Save dataset and optional metadata.
        
        Args:
            data: DataFrame to save
            filename: Output filename (without extension)
            metadata: Optional metadata dictionary
        """
        # Save CSV
        csv_path = self.base_path / f"{filename}.csv"
        data.to_csv(csv_path, index=False)
        print(f"Saved dataset: {csv_path}")
        
        # Save metadata if provided
        if metadata:
            self.metadata[filename] = metadata
            metadata_path = self.base_path / f"{filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Saved metadata: {metadata_path}")
    
    def save_all_metadata(self, filename: str = "dataset_info.json") -> None:
        """Save combined metadata for all datasets."""
        if self.metadata:
            metadata_path = self.base_path / filename
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            print(f"Saved combined metadata: {metadata_path}")


def main():
    """Demonstration of the strange attractor dataset generation."""
    
    print("Strange Attractor Dataset Generator")
    print("==================================")
    
    # Initialize components
    lorenz_gen = LorenzGenerator()
    embedding = TimeDelayEmbedding()
    dataset_mgr = DatasetManager()
    
    print("\n1. Generating single Lorenz trajectory...")
    single_traj = lorenz_gen.generate_single_trajectory()
    dataset_mgr.save_dataset(
        single_traj,
        "lorenz_single_run",
        {
            "description": "Single Lorenz-63 trajectory",
            "system": "Lorenz-63",
            "parameters": {"sigma": 10.0, "rho": 28.0, "beta": 8.0/3.0},
            "initial_state": [1.0, 1.0, 1.0],
            "time_span": [0, 40],
            "n_points": 10000
        }
    )
    
    print("\n2. Generating parameter sweep dataset...")
    rho_values = np.linspace(10, 40, 16)
    sweep_data = lorenz_gen.generate_parameter_sweep("rho", rho_values)
    dataset_mgr.save_dataset(
        sweep_data,
        "lorenz_parameter_sweep", 
        {
            "description": "Lorenz-63 system with varying rho parameter",
            "system": "Lorenz-63",
            "swept_parameter": "rho",
            "parameter_range": [float(rho_values.min()), float(rho_values.max())],
            "n_parameter_values": len(rho_values),
            "fixed_parameters": {"sigma": 10.0, "beta": 8.0/3.0},
            "initial_state": [1.0, 1.0, 1.0],
            "time_span": [0, 40],
            "n_points_per_trajectory": 10000,
            "total_points": len(sweep_data)
        }
    )
    
    print("\n3. Generating embedded signal dataset...")
    # Create sample signal
    t, signal = embedding.create_sample_signal()
    
    # Embed the signal
    tau = 20
    embedding_dim = 3
    embedded_coords = embedding.embed_signal(signal, tau, embedding_dim)
    
    # Convert to DataFrame
    embedded_df = pd.DataFrame(embedded_coords, columns=[f'x{i+1}' for i in range(embedding_dim)])
    
    dataset_mgr.save_dataset(
        embedded_df,
        "sample_signal_embedded",
        {
            "description": "Time-delay embedded sample signal",
            "source": "Synthetic sine wave with noise",
            "original_signal": {
                "duration_s": 10.0,
                "sampling_rate_hz": 1000.0,
                "frequency_hz": 5.0,
                "noise_level": 0.2
            },
            "embedding": {
                "tau": tau,
                "embedding_dimension": embedding_dim,
                "embedded_points": len(embedded_coords)
            }
        }
    )
    
    # Save combined metadata
    dataset_mgr.save_all_metadata()
    
    print(f"\n✓ All datasets saved to: {dataset_mgr.base_path}")
    print("✓ Ready for machine learning experiments!")


if __name__ == "__main__":
    main()