#!/usr/bin/env python
"""
hindmarsh_rose_dataset.py

Generates a collection of Hindmarsh–Rose neuronal trajectories across a
range of external currents (I_ext) and estimates the largest Lyapunov
exponent (LLE) for each.

Outputs
=======
1. Time‑series CSVs:  data/hr_I<value>.csv  (t, x, y, z)
2. Summary table:    data/hr_lyapunov_summary.csv  (I_ext, LLE)

Run:
    python hindmarsh_rose_dataset.py

Dependencies: numpy, pandas, matplotlib (optional for plots)
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
# Hindmarsh–Rose model                                                        #
###############################################################################

def hr_rhs(state, I_ext: float,
           a: float = 1.0, b: float = 3.0, c: float = 1.0, d: float = 5.0,
           r: float = 0.006, s: float = 4.0, x_rest: float = -1.6):
    """Right‑hand side of Hindmarsh–Rose equations."""
    x, y, z = state
    dx = y - a * x ** 3 + b * x ** 2 - z + I_ext
    dy = c - d * x ** 2 - y
    dz = r * (s * (x - x_rest) - z)
    return np.array([dx, dy, dz])


def rk4_step(state, dt, I_ext):
    k1 = hr_rhs(state, I_ext)
    k2 = hr_rhs(state + 0.5 * dt * k1, I_ext)
    k3 = hr_rhs(state + 0.5 * dt * k2, I_ext)
    k4 = hr_rhs(state + dt * k3, I_ext)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_hr(I_ext: float, dt: float = 0.05, total_time: float = 6_000,
                 burn_in: float = 1_000, init_state=(0.0, 0.0, 0.0)):
    """Integrate HR model and return array (N, 4) of t, x, y, z."""
    n_steps = int(total_time / dt)
    burn_steps = int(burn_in / dt)

    state = np.array(init_state, dtype=float)
    traj = np.empty((n_steps - burn_steps, 4), dtype=float)

    t = 0.0
    for i in range(n_steps):
        state = rk4_step(state, dt, I_ext)
        t += dt
        if i >= burn_steps:
            traj[i - burn_steps] = (t, *state)
    return traj

###############################################################################
# Lyapunov exponent estimator                                                 #
###############################################################################

def estimate_lle(I_ext: float, dt: float = 0.05, total_time: float = 8_000,
                 renorm_interval: int = 20, delta0: float = 1e-7):
    """Approximate largest Lyapunov exponent for HR using two‑trajectory method."""
    n_steps = int(total_time / dt)

    state1 = np.array([0.0, 0.0, 0.0])
    rand_dir = np.random.randn(3)
    rand_dir /= np.linalg.norm(rand_dir)
    state2 = state1 + delta0 * rand_dir

    sum_log = 0.0
    n_renorm = 0

    for step in range(n_steps):
        state1 = rk4_step(state1, dt, I_ext)
        state2 = rk4_step(state2, dt, I_ext)

        if (step + 1) % renorm_interval == 0:
            diff = state2 - state1
            dist = np.linalg.norm(diff)
            if dist == 0:
                dist = 1e-16
            sum_log += math.log(dist / delta0)
            n_renorm += 1
            diff *= delta0 / dist
            state2 = state1 + diff

    T = dt * n_steps
    return sum_log / T

###############################################################################
# Generate dataset                                                            #
###############################################################################

def main():
    os.makedirs("data", exist_ok=True)

    # Sweep I_ext: transitions from quiescent to spiking/bursting to chaos
    I_values = np.linspace(1.0, 3.5, 26)  # 0.1 increments

    summary = []

    for I_ext in I_values:
        print(f"→ Integrating HR for I_ext = {I_ext:.2f}")
        traj = integrate_hr(I_ext)
        df = pd.DataFrame(traj, columns=["t", "x", "y", "z"])
        csv_path = f"data/hr_I{I_ext:.2f}.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved {csv_path} ({len(df):,} samples)")

        # Optional quick plot for landmark I_ext values
        if HAS_PLOT and I_ext in (1.2, 2.2, 3.2):
            plt.figure(figsize=(10, 4))
            plt.plot(df["x"][:10_000])  # first 10k samples
            plt.title(f"Hindmarsh–Rose membrane potential (I_ext={I_ext:.2f})")
            plt.xlabel("sample"); plt.ylabel("x (mV)")
            plt.tight_layout()
            plot_path = f"data/hr_I{I_ext:.2f}_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved plot {plot_path}")

        print("    Estimating LLE…", end=" ")
        lle = estimate_lle(I_ext)
        print(f"LLE ≈ {lle:.4f}")
        summary.append({"I_ext": I_ext, "lyapunov": lle})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("data/hr_lyapunov_summary.csv", index=False)
    print("Saved summary → data/hr_lyapunov_summary.csv")

    if HAS_PLOT:
        plt.figure(figsize=(12, 6))
        plt.plot(summary_df["I_ext"], summary_df["lyapunov"], marker="o", linewidth=2, markersize=6)
        plt.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        plt.xlabel("I_ext (external current)")
        plt.ylabel("Approx. largest Lyapunov exp.")
        plt.title("Hindmarsh–Rose chaos map")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        chaos_map_path = "data/hr_chaos_map.png"
        plt.savefig(chaos_map_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved chaos map → {chaos_map_path}")


if __name__ == "__main__":
    main()