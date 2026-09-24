# EuCAIF whitened BBH dataset

Whitened two-detector (H1, L1) strain examples for binary-black-hole detection.
Each example is either **Gaussian noise only** (class 0) or **noise plus an
IMRPhenomPv2 signal** (class 1). Produced by `make_whitened_examples.py` with
[sage](https://github.com/MelissaLP/sage).

## At a glance

| Property | Value |
|---|---|
| Detectors | H1, L1 (in that order along axis 1) |
| Sample rate | 2048 Hz |
| Stored duration | 4 s per example (8192 samples) |
| Array shape | `x`: `(N, 2, 8192)` |
| Storage dtype | float32 by default (float64 optional) |
| Size per example | 64 KB (float32) / 128 KB (float64) |
| Classes | 50 % noise (0), 50 % signal + noise (1), shuffled |
| Noise | Stationary Gaussian, coloured by `aLIGOZeroDetHighPower` (both detectors) |
| Signal model | IMRPhenomPv2 (precessing spins), `ConstantProjection` onto each detector |
| Signal strength | Physical: set by the sampled luminosity distance (SNR rescaling optional) |
| Whitening | FIR inverse-spectrum truncation, equivalent to gwpy 3.0.14 `TimeSeries.whiten` |
| Merger time | `tc` uniform in [2.5, 3.5] s from the start of the 4 s window |

## How an example is made

1. **Noise.** Each example gets its own independent noise realisation,
   generated with `sage.data.noise.sample_synthetic_noise` over an 8 s window
   (4 s sample + 2 s padding on each side). The noise is coloured by
   `aLIGOZeroDetHighPower` and cut from a longer span, so it isn't periodic.
2. **Signal (class 1 only).** Source parameters are drawn from
   `sage_eucaif_waveform.yaml` (the prior's YAML is stored in the file
   attributes). IMRPhenomPv2 generates the projected strain in the frequency
   domain. The signal is converted to the time domain and added to the noise.
3. **Signal amplitude.** By default each signal keeps the physical amplitude
   set by its sampled luminosity distance, as sage's `IMRPhenomPv2` does
   without an `augment`. The amplitudes agree with PyCBC's IMRPhenomPv2 for
   the same parameters. The optimal SNR against the ASD that coloured the
   noise is stored for every signal. With `--snr-range MIN MAX`, each signal
   is instead rescaled to a network SNR drawn uniformly from that range, and
   its distance is updated to `distance / scale` so amplitude and distance
   stay consistent. That is the same convention as sage's
   `OptimalSNRRescaler` augment, but the distances no longer follow the
   prior.
4. **Whitening.** The full 8 s is whitened with `sage.dsp.whiten.FIRWhitening`
   using the known ASD: a 4 s FIR filter, 15 Hz highpass, Hann taper. That
   removes 2 s from each end, leaving exactly the central 4 s sample. Whitened
   noise has zero mean and unit variance.

## File layout (HDF5)

```
x                   (N, 2, 8192)  float32    whitened strain [example, detector, time] (float64 with --store-dtype float64)
t                   (8192,)       float64    time of each sample, s from window start
metadata/class      (N,)          int8       0 = noise only, 1 = signal + noise
metadata/tc         (N,)          float64    merger time, s from window start
metadata/mchirp     (N,)          float64    chirp mass, solar masses
metadata/ra         (N,)          float64    right ascension, rad, [0, 2pi)
metadata/dec        (N,)          float64    declination, rad, [-pi/2, pi/2]
metadata/gmst       (N,)          float64    sidereal time used for the projection, rad, [0, 2pi)
metadata/distance   (N,)          float64    luminosity distance, Mpc
metadata/snr        (N,)          float32    network optimal SNR
metadata/snr_det    (N, 2)        float32    per-detector optimal SNR (H1, L1)
```

- **Class 0:** `tc`, `mchirp`, `ra`, `dec`, `gmst`, `distance`, `snr` and `snr_det` are **NaN**.
- **Row alignment:** every array is aligned by row `i`, so `x[i]` goes with
  `metadata/*[i]`.
- **File attributes** (`f.attrs`) record the full configuration: sample rate,
  window and padding lengths, ASD names, whitening settings, SNR rescaling, the
  waveform-prior YAML, seed, sage commit and creation time.

### Metadata definitions

| Field | Meaning |
|---|---|
| `tc` | Geocentric coalescence time from the start of the stored window. The measured peak of the signal lands within about 0.1 s of `tc` (detector light-travel delay plus the model's own peak offset). |
| `mchirp` | Chirp mass in solar masses from the prior, (m1 m2)^(3/5) / (m1 + m2)^(1/5). No redshift is applied. Component masses are uniform in [10, 50] M☉, so `mchirp` is about 8.7–43.5 M☉. |
| `ra`, `dec` | Sky position in radians, isotropic prior. |
| `gmst` | Greenwich Mean Sidereal Time used to project the signal onto the detectors. sage's `ConstantProjection` draws it uniformly at random for each signal instead of deriving it from a GPS time. The detector response depends on `ra` only through the hour angle `gmst - ra`, so **`ra` alone can't be recovered from the data**. To estimate the sky position, regress the hour angle `(gmst - ra) mod 2π` (with `dec`) instead of `ra`. |
| `class` | Label for detection: 0 for noise, 1 for signal. |
| `distance` | Luminosity distance in Mpc. Derived by the prior from `chirp_distance` (uniform in volume, 130–350 Mpc) and `mchirp`, so heavier systems are placed farther away; it ranges from about 1 to 7 Gpc. It is updated when `--snr-range` is used. |
| `snr`, `snr_det` | Optimal SNRs of the injected signal, not matched-filter SNRs recovered from the noisy data. |

## Loading

```python
import h5py, numpy as np, torch

with h5py.File("eucaif_whitened.h5", "r") as f:
    x = torch.from_numpy(f["x"][:1000])                          # (1000, 2, 8192)
    y = torch.from_numpy(f["metadata/class"][:1000].astype(np.int64))
    meta = {k: f[f"metadata/{k}"][:1000] for k in ["tc", "mchirp", "ra", "dec"]}
    fs = f.attrs["sample_rate"]
```

h5py reads only the slices you ask for, so a dataset larger than memory can be
streamed batch by batch, for example with a `torch.utils.data.Dataset` that
indexes the file.

## Generating

```bash
python make_whitened_examples.py sage_eucaif_waveform.yaml \
    --n-per-class 25000 --out eucaif_whitened.h5
```

| Option | Default | Meaning |
|---|---|---|
| `--n-per-class` | 100 | examples per class; the file holds twice this |
| `--chunk-per-class` | 1000 | examples per class generated at once; peak RAM is about 1–2 GB at 1000 |
| `--store-dtype` | float32 | `float64` doubles the size and also whitens in float64 |
| `--snr-range MIN MAX` | off | rescale to a network optimal SNR uniform in [MIN, MAX]; off keeps physical distances |
| `--seed` | 150914 | reproducibility; the same seed gives the same file (up to floating-point differences between machines) |

Before writing, the script runs a quick check and saves a plot
(`whitened_examples.png`). It stops with an error if the data config is
inconsistent or if any merger falls outside the window.

**Storage** (2 detectors × 8192 samples):

| Examples (both classes) | float32 | float64 |
|---|---|---|
| 10,000 | 0.66 GB | 1.3 GB |
| 38,000 | 2.5 GB | 5.0 GB |
| 76,000 | 5.0 GB | 10 GB |

## Caveats

- **Many signals are weak (physical distances, the default).** With this
  prior and aLIGO design sensitivity, the network optimal SNR of the signals
  has median about 5 (10–90 % range: 2–10). About 50 % have SNR ≥ 5, 19 %
  have SNR ≥ 8 and 4 % have SNR ≥ 12. A large fraction of class-1 examples
  are therefore practically indistinguishable from noise. Use `metadata/snr`
  to weight, filter or evaluate by SNR bin, or generate with `--snr-range`
  for training.
- **Idealised noise.** It is stationary and Gaussian, from one analytic PSD,
  with no glitches, spectral lines or drift. Performance on real detector data
  will be lower.
- **Idealised whitening.** It uses the exact ASD that coloured the noise. On
  real data the ASD has to be estimated, and the whitening is less perfect.
- **float64.** It whitens and stores in float64, which matches gwpy to about 1e-8
  instead of 1e-6 for float32. The waveforms themselves are generated in
  float32 (sage's `cfg.dtype`), so float64 improves the whitening, not the
  signal model. It's mainly useful for validation; float32 is plenty for
  training.
- **Long inspirals.** The lightest systems (about 10 + 10 M☉) are longer than
  the 8 s generation window from 20 Hz. The earliest part of their inspiral
  wraps to the end of the window and falls mostly in the whitening padding
  that is discarded.
- **Shuffling.** Examples are shuffled within each generation chunk
  (`--chunk-per-class` per class), not across the whole file. Shuffle again
  when training.
