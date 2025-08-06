#!/usr/bin/env python
"""
rossler_advanced_dataset.py

Creates an "advanced" strange‑attractor dataset based on the Rössler
system *and* computes an approximate largest Lyapunov exponent (LLE) for
each parameter setting.

The script will:
  1. Sweep the Rössler control parameter `c` across a range (default 4 → 8).
  2. For each `c`, integrate the Rössler ODE for a single long trajectory and
     save the coordinates to ./data/rossler_cXX.csv (where XX is the c value).
  3. Compute a finite‑time LLE estimate via the classic two‑trajectory‑with‑
     renormalisation method and write those numbers to
     ./data/rossler_lyapunov_summary.csv.

Result = **a labelled benchmark where you know how chaotic each series is**.

Run it like:
    python rossler_advanced_dataset.py

Dependencies (install if missing):
    numpy pandas matplotlib (optional, for quick plots)
"""

import os
import math
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

###############################################################################
# Rössler ODE definitions                                                     #
###############################################################################

def rossler_rhs(state, a: float = 0.2, b: float = 0.2, c: float = 5.7):
    """Rössler right‑hand side."""
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])


def integrate_rossler(
    c: float,
    a: float = 0.2,
    b: float = 0.2,
    dt: float = 0.01,
    n_steps: int = 110_000,
    burn_in: int = 10_000,
    init_state=(1.0, 0.0, 0.0),
):
    """Integrate the Rössler system using 4th‑order Runge‑Kutta.

    Returns an (N, 4) array with columns t, x, y, z (post burn‑in).
    """

    def rk4_step(state):
        k1 = rossler_rhs(state, a, b, c)
        k2 = rossler_rhs(state + 0.5 * dt * k1, a, b, c)
        k3 = rossler_rhs(state + 0.5 * dt * k2, a, b, c)
        k4 = rossler_rhs(state + dt * k3, a, b, c)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    state = np.array(init_state, dtype=float)
    traj = np.empty((n_steps - burn_in, 4), dtype=float)

    for i in range(n_steps):
        state = rk4_step(state)
        if i >= burn_in:
            traj[i - burn_in, 0] = (i - burn_in) * dt  # time
            traj[i - burn_in, 1:] = state  # x, y, z

    return traj  # shape (N, 4)

###############################################################################
# Lyapunov exponent estimation                                               #
###############################################################################

def estimate_lyapunov(
    c: float,
    a: float = 0.2,
    b: float = 0.2,
    dt: float = 0.01,
    n_steps: int = 120_000,
    renorm_interval: int = 10,
    init_state=(1.0, 0.0, 0.0),
    delta0: float = 1e-8,
):
    """Approximate the largest Lyapunov exponent using two‑trajectory method.

    The algorithm:
      • Start two states separated by delta0 in a random direction.
      • Integrate both systems in parallel.
      • Every `renorm_interval` steps, measure the separation d, accumulate
        ln(d/delta0), then renormalise to delta0.
      • LLE ≈ (1 / (T_total)) * Σ ln(d/delta0).
    """

    def rk4_step(state):
        k1 = rossler_rhs(state, a, b, c)
        k2 = rossler_rhs(state + 0.5 * dt * k1, a, b, c)
        k3 = rossler_rhs(state + 0.5 * dt * k2, a, b, c)
        k4 = rossler_rhs(state + dt * k3, a, b, c)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Primary and perturbed trajectories
    state1 = np.array(init_state, dtype=float)
    # Choose a random unit vector for initial perturbation
    rand_dir = np.random.randn(3)
    rand_dir /= np.linalg.norm(rand_dir)
    state2 = state1 + delta0 * rand_dir

    sum_ln = 0.0
    n_renorm = 0

    for step in range(n_steps):
        state1 = rk4_step(state1)
        state2 = rk4_step(state2)

        if (step + 1) % renorm_interval == 0:
            diff = state2 - state1
            dist = np.linalg.norm(diff)
            if dist == 0:
                dist = 1e-16  # avoid log(0)
            sum_ln += math.log(dist / delta0)
            n_renorm += 1
            # Renormalise separation to delta0
            diff *= delta0 / dist
            state2 = state1 + diff

    T_total = dt * n_steps
    return sum_ln / (T_total)  # LLE estimate

###############################################################################
# Main driver                                                                 #
###############################################################################

def main():
    os.makedirs("data", exist_ok=True)

    # Parameter sweep for c (note: chaos around c ≈ 5.7)
    c_values = np.linspace(4.0, 8.0, 17)  # 17 values at 0.25 spacing

    summary_rows = []

    for c in c_values:
        print(f"Integrating Rössler for c = {c:.2f}…")
        traj = integrate_rossler(c)

        # Save trajectory to CSV
        df = pd.DataFrame(traj, columns=["t", "x", "y", "z"])
        csv_path = f"data/rossler_c{c:.2f}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  → Saved {csv_path} ({len(df):,} samples)")

        # Quick plot for the first few parameter values (optional)
        if HAS_PLOT and c in (4.0, 5.0, 5.7, 7.0):
            plt.figure(figsize=(8, 6))
            plt.plot(df["x"], df["z"], linewidth=0.3)
            plt.title(f"Rössler attractor slice (c = {c:.2f})")
            plt.xlabel("x"); plt.ylabel("z")
            plt.tight_layout()
            plot_path = f"data/rossler_c{c:.2f}_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  → Saved plot {plot_path}")

        # Estimate Lyapunov exponent (might take ~2–3 seconds each)
        print("  Estimating Lyapunov exponent…", end=" ")
        lle = estimate_lyapunov(c)
        print(f"LLE ≈ {lle:.4f}")

        summary_rows.append({"c": c, "lyapunov": lle})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = "data/rossler_lyapunov_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table → {summary_path}")

    if HAS_PLOT:
        plt.figure(figsize=(10, 6))
        plt.plot(summary_df["c"], summary_df["lyapunov"], marker="o", linewidth=2, markersize=6)
        plt.axhline(0, color="red", linewidth=1, linestyle="--", alpha=0.7)
        plt.xlabel("c parameter")
        plt.ylabel("Approx. largest Lyapunov exponent")
        plt.title("Rössler chaos map (positive = chaotic)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        chaos_plot_path = "data/rossler_chaos_map.png"
        plt.savefig(chaos_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved chaos map → {chaos_plot_path}")


if __name__ == "__main__":
    main()