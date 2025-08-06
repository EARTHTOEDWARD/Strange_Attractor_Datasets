#!/usr/bin/env python
"""
test_embed_eeg.py

Test script for embed_eeg_chbmit.py functionality without requiring actual EEG data.
Creates synthetic EEG-like signals and demonstrates the embedding functionality.
"""

import numpy as np
import pandas as pd
import os

# Import embedding function directly from the main script
import sys
sys.path.append('.')

def create_synthetic_eeg(duration_sec: float = 10.0, sfreq: float = 256.0, 
                        include_seizure: bool = True) -> tuple:
    """Create synthetic EEG-like signal with optional seizure-like activity."""
    
    n_samples = int(duration_sec * sfreq)
    t = np.linspace(0, duration_sec, n_samples)
    
    # Base EEG-like signal: combination of multiple frequency components
    signal = (
        0.3 * np.sin(2 * np.pi * 8 * t) +      # Alpha rhythm
        0.2 * np.sin(2 * np.pi * 13 * t) +     # Beta rhythm  
        0.1 * np.sin(2 * np.pi * 4 * t) +      # Theta rhythm
        0.1 * np.random.randn(n_samples)       # Background noise
    )
    
    seizure_intervals = []
    
    if include_seizure:
        # Add seizure-like high amplitude, high frequency activity
        seizure_start = duration_sec * 0.4  # Start at 40% of recording
        seizure_end = duration_sec * 0.6    # End at 60% of recording
        
        seizure_start_idx = int(seizure_start * sfreq)
        seizure_end_idx = int(seizure_end * sfreq)
        
        # High frequency, high amplitude oscillation during "seizure"
        seizure_t = t[seizure_start_idx:seizure_end_idx]
        seizure_component = 2.0 * np.sin(2 * np.pi * 25 * seizure_t)  # 25 Hz
        
        signal[seizure_start_idx:seizure_end_idx] += seizure_component
        seizure_intervals = [(seizure_start, seizure_end)]
    
    return signal, seizure_intervals, sfreq

def embed_signal_test(signal: np.ndarray, tau: int, m: int) -> np.ndarray:
    """Test version of embed_signal function."""
    N = len(signal) - (m - 1) * tau
    if N <= 0:
        raise ValueError("Signal too short for given tau and m.")
    emb = np.empty((N, m), dtype=np.float32)
    for i in range(m):
        emb[:, i] = signal[i * tau : i * tau + N]
    return emb

def test_embedding():
    """Test the EEG embedding functionality."""
    
    print("Creating synthetic EEG signal...")
    signal, seizure_intervals, sfreq = create_synthetic_eeg(duration_sec=20.0)
    
    print(f"Signal length: {len(signal)} samples")
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Duration: {len(signal)/sfreq:.1f} seconds")
    
    if seizure_intervals:
        print(f"Seizure intervals: {seizure_intervals}")
    
    # Test embedding parameters
    tau = int(0.02 * sfreq)  # 20ms delay in samples
    m = 3  # 3D embedding
    
    print(f"\nEmbedding parameters:")
    print(f"  tau = {tau} samples ({tau/sfreq*1000:.1f} ms)")
    print(f"  dimension = {m}")
    
    # Create segments based on seizure intervals
    segments = []
    if seizure_intervals:
        for start_sec, end_sec in seizure_intervals:
            start_samp = int(start_sec * sfreq)
            end_samp = int(end_sec * sfreq)
            segments.append((start_samp, end_samp, 1, "seizure"))
        
        # Add baseline segments
        prev = 0
        for start_samp, end_samp, _, _ in segments:
            if prev < start_samp:
                segments.append((prev, start_samp, 0, "baseline"))
            prev = end_samp
        if prev < len(signal):
            segments.append((prev, len(signal), 0, "baseline"))
    else:
        segments = [(0, len(signal), 0, "baseline")]
    
    # Process each segment
    os.makedirs("data/embedding_eeg", exist_ok=True)
    
    for idx, (start, end, label, label_name) in enumerate(sorted(segments)):
        segment_signal = signal[start:end]
        
        print(f"\nProcessing {label_name} segment {idx}:")
        print(f"  Samples: {start} to {end} ({len(segment_signal)} total)")
        print(f"  Duration: {len(segment_signal)/sfreq:.2f} seconds")
        
        try:
            emb = embed_signal_test(segment_signal, tau, m)
            print(f"  Embedded shape: {emb.shape}")
            
            # Save to CSV
            df = pd.DataFrame(emb, columns=[f"x{i}" for i in range(m)])
            df["label"] = label
            
            out_path = f"data/embedding_eeg/synthetic_test_{label_name}_{idx}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
            
        except ValueError as e:
            print(f"  Error: {e}")
    
    print("\nTest completed successfully!")
    print("\nTo use with real CHB-MIT data:")
    print("1. Install MNE-Python: pip install mne")
    print("2. Download CHB-MIT dataset from: https://physionet.org/content/chbmit/")  
    print("3. Run: python embed_eeg_chbmit.py --edf path/to/file.edf --channel F7")

if __name__ == "__main__":
    test_embedding()