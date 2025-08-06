#!/usr/bin/env python
"""
chaotic_dataset_example.py

A self‑contained script that demonstrates two things:

1. Generates a parameter‑sweep dataset of Lorenz‑63 trajectories and saves
   them to ./data/lorenz_parameter_sweep.csv.
2. Creates a simple noisy sine‑wave signal, converts it to a 3‑D delay‑
   coordinate embedding (a reconstructed attractor), and saves the result
   to ./data/synthetic_signal_embedded.csv.

Run the script from a terminal:
    python chaotic_dataset_example.py

Requirements (install via pip if needed):
    numpy scipy pandas matplotlib

Feel free to tweak the parameters at the bottom of the file (rho_values,
embedding tau, etc.) and re‑run to create new datasets.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Matplotlib is optional; if missing the script still creates the datasets.
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:  # pragma: no cover
    HAS_PLOT = False

###############################################################################
# PART 1: Lorenz‑63 synthetic chaotic data                                    #
###############################################################################

def lorenz(t, state, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3):
    """Right‑hand side of the Lorenz‑63 ODE system."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def generate_lorenz_dataset(
    rho_values,
    init_state=(1.0, 1.0, 1.0),
    t_span=(0.0, 40.0),
    samples_per_run: int = 10_000,
):
    """Integrate the Lorenz system for each rho and stack all trajectories."""
    rows = []
    t_eval = np.linspace(t_span[0], t_span[1], samples_per_run)

    for rho in rho_values:
        sol = solve_ivp(
            lorenz,
            t_span,
            init_state,
            args=(10.0, rho, 8 / 3),  # sigma, rho, beta
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-12,
        )

        df = pd.DataFrame(
            {
                "t": sol.t,
                "x": sol.y[0],
                "y": sol.y[1],
                "z": sol.y[2],
                "rho": rho,  # label describing parameter value
            }
        )
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


###############################################################################
# PART 2: Delay‑coordinate embedding for real‑world (or synthetic) signals    #
###############################################################################

def embed_signal(signal: np.ndarray, tau: int = 20, m: int = 3):
    """Return an m‑dimensional delay embedding of a 1‑D signal.

    Each row is (x(t), x(t+tau), x(t+2*tau), ..., x(t+(m-1)*tau)).
    """
    N = len(signal) - (m - 1) * tau
    if N <= 0:
        raise ValueError("Time series is too short for the chosen tau and m.")

    embedded = np.zeros((N, m))
    for i in range(m):
        embedded[:, i] = signal[i * tau : i * tau + N]

    columns = [f"x{i}" for i in range(m)]
    return pd.DataFrame(embedded, columns=columns)


def create_synthetic_signal(duration: float = 10.0, fs: int = 1_000):
    """Return a noisy 5‑Hz sine wave of given duration and sample rate."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(len(t))
    return signal, fs


###############################################################################
# MAIN SCRIPT                                                                #
###############################################################################

def main():
    # Ensure an output directory exists
    os.makedirs("data", exist_ok=True)

    # 1. Generate Lorenz dataset ------------------------------------------------
    rho_values = np.linspace(10, 40, 16)  # 16 evenly spaced rho settings
    lorenz_df = generate_lorenz_dataset(rho_values)
    lorenz_path = "data/lorenz_parameter_sweep.csv"
    lorenz_df.to_csv(lorenz_path, index=False)
    print(f"Saved Lorenz dataset → {lorenz_path}")

    # Optional quick visual check (slice x vs z for rho=28)
    if HAS_PLOT:
        slice_df = lorenz_df[lorenz_df["rho"] == 28.0]
        plt.plot(slice_df["x"], slice_df["z"], linewidth=0.3)
        plt.title("Lorenz attractor slice (rho = 28)")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.tight_layout()
        plt.savefig("data/lorenz_attractor_slice.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved Lorenz visualization → data/lorenz_attractor_slice.png")

    # 2. Create and embed a synthetic signal -----------------------------------
    signal, fs = create_synthetic_signal()
    tau_samples = int(0.02 * fs)  # 20 ms delay expressed in samples
    embedded_df = embed_signal(signal, tau=tau_samples, m=3)
    embed_path = "data/synthetic_signal_embedded.csv"
    embedded_df.to_csv(embed_path, index=False)
    print(f"Saved embedded signal → {embed_path}")

    if HAS_PLOT:
        plt.scatter(embedded_df["x0"], embedded_df["x1"], s=0.3)
        plt.title("Delay‑embedded signal (x(t) vs x(t+τ))")
        plt.xlabel("x(t)")
        plt.ylabel("x(t+τ)")
        plt.tight_layout()
        plt.savefig("data/embedded_signal_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved embedding visualization → data/embedded_signal_plot.png")


if __name__ == "__main__":
    main()