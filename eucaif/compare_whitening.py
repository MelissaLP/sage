"""
Compare sage's FIRWhitening (torch) against gwpy 3.0.14 TimeSeries.whiten
on synthetic H1/L1 noise coloured by aLIGOZeroDetHighPower.

Changes vs. your original script are marked  # CHANGED  /  # NEW.
"""
import time

import matplotlib.pyplot as plt   # NEW: was used below but never imported
import numpy as np
import pandas as pd
import torch
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

from sage.data.noise import sample_synthetic_noise
from sage.dsp.whiten import FIRWhitening


def to_numpy(x):
    """Convert a torch tensor (or an already-numpy array) to numpy."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def asd_to_frequencyseries(asd_1d: np.ndarray, sample_rate: float, eps: float) -> FrequencySeries:
    """
    Wrap a 1-D ASD (assumed to span 0 -> Nyquist evenly) as a gwpy
    FrequencySeries, with invalid (<= eps) bins pushed to a large value
    so gwpy's un-guarded ``1 / asd`` doesn't blow up to inf/nan --
    mirrors FIRWhitening._invert_asd's excise-not-amplify behaviour.

    df is derived from the ASD's OWN bin count -- an ASD with n_freq bins
    spanning 0 Hz to Nyquist has df = (sample_rate / 2) / (n_freq - 1).
    """
    n_freq = asd_1d.shape[-1]
    df = sample_rate / (2 * (n_freq - 1))
    asd_safe = asd_1d.astype(np.float64).copy()
    invalid = asd_safe <= eps
    if invalid.any():
        sentinel = asd_safe[~invalid].max() * 1e6 if (~invalid).any() else 1e30
        asd_safe[invalid] = sentinel
    return FrequencySeries(asd_safe, df=df, f0=0)


def compare_whitening(
    n_trials: int,
    sample_length_in_s: float,
    sample_rate: float = 2048.0,
    fduration: float = 2.0,
    window: str = "hann",
    eps: float = None,
    highpass: float = None,
    dtype: torch.dtype = torch.float32,   # NEW: torch.float64 for a like-for-like check vs gwpy
    seed: int = 0,                        # NEW: reproducible noise
) -> pd.DataFrame:
    """
    Generate `n_trials` synthetic noise realisations for H1/L1, whiten each
    with both FIRWhitening (torch, one batched call) and gwpy's
    TimeSeries.whiten (loop), and compute the gwpy-vs-torch difference
    statistics per (trial, detector) over the valid, non-edge region.

    NOTE: fftlength/overlap were removed -- gwpy only uses them to
    *estimate* an ASD when ``asd=None``. With ``asd=`` given they are ignored.
    """
    noise_batch, asd_used = sample_synthetic_noise(
        sample_length_in_s,
        ["aLIGOZeroDetHighPower", "aLIGOZeroDetHighPower"],
        detectors=["H1", "L1"],
        batch=n_trials,
        sample_rate=sample_rate,          # CHANGED: pass explicitly so everything agrees
        seed=seed,
        dtype=dtype,
        is_asd=True,
    )  # noise_batch: (n_trials, D, T); asd_used: (D, F_in)
    noise_batch = noise_batch.to(dtype)
    asd_used = torch.as_tensor(asd_used).to(dtype)

    T = noise_batch.shape[-1]
    n_detectors = noise_batch.shape[1]

    # CHANGED: use the function arguments (were hard-coded 2048 / "hann" / 2.0).
    fir_whitener = FIRWhitening(
        sample_rate=sample_rate, window=window, fduration=fduration,
        eps=eps, highpass=highpass,
        dtype=dtype,                       # NEW: float32 (fast) or float64 (validation)
    )
    # CHANGED: seq_len must be an int (sample_length_in_s * sample_rate is a float),
    # and it must equal the real input length T.
    fir_whitener.set_asd(asd_used, seq_len=T)   # designs the FIR filter ONCE, here

    eps_used = (
        fir_whitener.eps if fir_whitener.eps is not None
        else torch.finfo(dtype).tiny
    )

    # -- torch: whiten every trial and detector in ONE batched call --
    # CHANGED: call WITHOUT asd= so the cached filter is used. Passing asd=
    # (as before) silently redesigns the filter on every call.
    t0 = time.perf_counter()
    out_all = fir_whitener(noise_batch)                       # (n_trials, D, T_valid)
    t_torch = time.perf_counter() - t0

    # NEW: sanity check -- cached path == per-call-design path
    out_percall = fir_whitener(noise_batch, asd=asd_used)
    cache_err = (out_all - out_percall).abs().max() / out_all.std()
    print(f"cached vs per-call design: max|diff|/std = {cache_err:.2e}")

    pad = fir_whitener.pad

    asd_fs_per_detector = [
        asd_to_frequencyseries(to_numpy(asd_used[det]), sample_rate, eps_used)
        for det in range(n_detectors)
    ]

    records = []
    t_gwpy = 0.0
    for det in range(n_detectors):
        asd_fs = asd_fs_per_detector[det]
        for trial in range(n_trials):
            x_np = to_numpy(noise_batch[trial, det]).astype(np.float64)
            ts = TimeSeries(x_np, sample_rate=sample_rate)

            t0 = time.perf_counter()
            gwpy_out = ts.whiten(
                window=window, detrend="constant", asd=asd_fs,
                fduration=fduration, highpass=highpass,   # CHANGED: keep in sync with torch
            )
            t_gwpy += time.perf_counter() - t0

            gwpy_valid = gwpy_out.value[pad:T - pad]
            torch_valid = to_numpy(out_all[trial, det])
            diff = gwpy_valid - torch_valid

            records.append({
                "trial": trial,
                "detector": det,
                "mean": diff.mean(),
                "std": diff.std(),
                "mean_abs": np.abs(diff).mean(),
                "max_abs": np.abs(diff).max(),
                # NEW: errors relative to the whitened signal's own size
                "rel_max_abs": np.abs(diff).max() / gwpy_valid.std(),
                "gwpy_std": gwpy_valid.std(),
                "torch_std": torch_valid.std(),
            })

            if (trial + 1) % 100 == 0:
                print(f"detector {det}: {trial + 1}/{n_trials} trials done")

    print(f"timing: torch batched {t_torch * 1e3:.1f} ms total | "
          f"gwpy loop {t_gwpy * 1e3:.1f} ms total "
          f"({n_trials * n_detectors} series)")
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    # CHANGED: data_cfg was undefined here. Use the registered config if
    # there is one, else fall back to 16 s.
    try:
        from sage.core.config import get_data_cfg
        sample_length_in_s = get_data_cfg().sample_length_in_s
    except Exception:
        sample_length_in_s = 16.0

    df = compare_whitening(n_trials=20, sample_length_in_s=sample_length_in_s)
    print(df.head())
    print(df[["mean", "std", "mean_abs", "max_abs", "rel_max_abs", "gwpy_std", "torch_std"]].describe())

    # CHANGED: plotting moved inside __main__ (df only exists here)
    cols = ["mean", "std", "mean_abs", "max_abs"]
    colors = ["seagreen", "steelblue", "darkorange", "crimson"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col], alpha=0.7, color=colors[i], bins=100)
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()
