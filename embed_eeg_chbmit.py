#!/usr/bin/env python
"""
embed_eeg_chbmit.py
===================

Convert raw CHB‑MIT scalp‑EEG signals into delay‑coordinate embeddings so that
each 1‑D channel becomes a 3‑D (or higher‑D) attractor trajectory suitable for
reservoir computing, topological‐data‑analysis, etc.

Key features
------------
* **Reads .edf files with MNE‑Python** (handles metadata, sampling rate, etc.).
* **Embeds any chosen channel** using Takens' delay embedding with parameters 
  `tau` (delay in *samples*) and `m` (dimension).
* **Optional segmentation by seizure annotations** if you have a CSV listing
  start/end times (in seconds) for seizures in the same .edf record.
* **Outputs one CSV file per segment** under ./data/embedding_eeg/ with columns
  `x0, x1, …, x{m-1}, label` where label = 1 for seizure, 0 otherwise.

Usage example
-------------
    python embed_eeg_chbmit.py \
        --edf raw_eeg/chb01/chb01_03.edf \
        --channel F7 \
        --tau 30 \
        --dim 3 \
        --seizure-csv raw_eeg/chb01/chb01_03_seizures.csv

Dependencies
------------
    pip install mne numpy pandas

The script will *not* download CHB‑MIT automatically—it expects you to place
.edf files inside a directory (e.g. `raw_eeg/`). You can grab them from
https://physionet.org/content/chbmit/.
"""

import os
import argparse
from typing import Optional, List

import numpy as np
import pandas as pd
import mne

###############################################################################
# Helpers                                                                     #
###############################################################################

def embed_signal(signal: np.ndarray, tau: int, m: int) -> np.ndarray:
    """Return delay‐embedded matrix of shape (N, m)."""
    N = len(signal) - (m - 1) * tau
    if N <= 0:
        raise ValueError("Signal too short for given tau and m.")
    emb = np.empty((N, m), dtype=np.float32)
    for i in range(m):
        emb[:, i] = signal[i * tau : i * tau + N]
    return emb


def load_seizure_intervals(csv_path: str) -> List[tuple]:
    """Read a CSV with `start_sec,end_sec` rows. Return list of (start, end)."""
    df = pd.read_csv(csv_path)
    return [(float(row["start_sec"]), float(row["end_sec"])) for _, row in df.iterrows()]


###############################################################################
# Main routine                                                                #
###############################################################################

def process_edf(edf_path: str, channel: str, tau: int, m: int,
                seizure_csv: Optional[str] = None, out_dir: str = "data/embedding_eeg"):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {edf_path} …")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    if channel not in raw.ch_names:
        raise ValueError(f"Channel '{channel}' not found. Available: {raw.ch_names}")

    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=[channel])[0]  # shape (n_samples,)
    total_samples = len(data)

    # Determine segments (whole file if no seizure annotation)
    if seizure_csv:
        intervals = load_seizure_intervals(seizure_csv)
        segments = []
        for start_sec, end_sec in intervals:
            start_samp = int(start_sec * sfreq)
            end_samp = int(end_sec * sfreq)
            segments.append((start_samp, end_samp, 1))  # label 1 = seizure
        # Non‑seizure segments (simple complement set)
        prev = 0
        for start_samp, end_samp, _ in segments:
            if prev < start_samp:
                segments.append((prev, start_samp, 0))
            prev = end_samp
        if prev < total_samples:
            segments.append((prev, total_samples, 0))
    else:
        segments = [(0, total_samples, 0)]

    # Process each segment
    for idx, (start, end, label) in enumerate(sorted(segments)):
        segment_signal = data[start:end]
        try:
            emb = embed_signal(segment_signal, tau, m)
        except ValueError as e:
            print(f"Skipping segment {idx}: {e}")
            continue

        df = pd.DataFrame(emb, columns=[f"x{i}" for i in range(m)])
        df["label"] = label

        base = os.path.basename(edf_path).replace(".edf", "")
        label_str = "seiz" if label == 1 else "base"
        out_path = os.path.join(out_dir, f"{base}_{channel}_{label_str}_{idx}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}  ({len(df):,} rows)")

    print("Done.")


###############################################################################
# CLI entry‐point                                                             #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Embed CHB‑MIT EEG channel into phase space.")
    parser.add_argument("--edf", required=True, help="Path to .edf file (CHB‑MIT record).")
    parser.add_argument("--channel", required=True, help="EEG channel name (e.g. F7).")
    parser.add_argument("--tau", type=int, default=20, help="Delay in samples (default 20).")
    parser.add_argument("--dim", type=int, default=3, help="Embedding dimension m (default 3).")
    parser.add_argument("--seizure-csv", help="Optional CSV of seizure intervals (start_sec,end_sec).")
    parser.add_argument("--out-dir", default="data/embedding_eeg", help="Output directory for CSVs.")
    args = parser.parse_args()

    process_edf(args.edf, args.channel, args.tau, args.dim, args.seizure_csv, args.out_dir)


if __name__ == "__main__":
    main()