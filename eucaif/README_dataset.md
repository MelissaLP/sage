# EuCAIF whitened BBH dataset

Whitened two-detector (H1, L1) strain examples for binary-black-hole detection.
Each example is either **Gaussian noise only** (class 0) or **noise plus a
non-spinning IMRPhenomPv2 signal** (class 1). Produced by
`make_whitened_examples.py` with [sage](https://github.com/MelissaLP/sage).
The source population and window follow the ggwd / ML-challenge setup; see
"Relation to the ggwd setup" below.

## At a glance

| Property | Value |
|---|---|
| Detectors | H1, L1 (in that order along axis 1) |
| Sample rate | 2048 Hz |
| Stored duration | 2 s per example (4096 samples) |
| Array shape | `x`: `(N, 2, 4096)` |
| Storage dtype | float32 by default (float64 optional) |
| Size per example | 32 KB (float32) / 64 KB (float64) |
| Classes | 50 % noise (0), 50 % signal + noise (1), shuffled |
| Noise | Stationary Gaussian, coloured by `aLIGOZeroDetHighPower` (both detectors) |
| Signal model | IMRPhenomPv2, non-spinning; `ConstantProjection` onto each detector |
| Signal strength | Physical: set by the sampled luminosity distance; `chirp_distance` 65–175 Mpc, median network SNR ≈ 10 |
| Whitening | 4 s FIR inverse-spectrum truncation, equivalent to gwpy 3.0.14 `TimeSeries.whiten` |
| Merger time | `tc` uniform in [0.5, 1.5] s from the start of the 2 s window |

## How an example is made

1. **Noise.** Each example gets its own independent noise realisation,
   generated with `sage.data.noise.sample_synthetic_noise` over a 10 s window
   (2 s sample + 4 s padding on each side). The noise is coloured by
   `aLIGOZeroDetHighPower` and cut from a longer span, so it isn't periodic.
2. **Signal (class 1 only).** Source parameters are drawn from
   `sage_eucaif_waveform.yaml` (the prior's YAML is stored in the file
   attributes). IMRPhenomPv2 generates the projected strain in the frequency
   domain on the 10 s grid. The signal is converted to the time domain and
   added to the noise. The 4 s padding is long enough that even the longest
   signals (10 + 10 M☉, about 6 s from 20 Hz) fit in the window without
   wrapping around.
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
4. **Whitening.** The full 10 s is whitened with `sage.dsp.whiten.FIRWhitening`
   using the known ASD: a 4 s FIR filter (as ggwd's
   `whitening_max_filter_duration`), 15 Hz highpass, Hann taper. The whitened
   series is then cropped to the central 2 s sample, which is well clear of
   the 2 s the filter corrupts at each end. Whitened noise has zero mean and
   unit variance.

## File layout (HDF5)

```
x                   (N, 2, 4096)  float32    whitened strain [example, detector, time] (float64 with --store-dtype float64)
t                   (4096,)       float64    time of each sample, s from window start
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
| `distance` | Luminosity distance in Mpc. Derived by the prior from `chirp_distance` (uniform in volume, 65–175 Mpc) and `mchirp`, so heavier systems are placed farther away; it ranges from about 0.5 to 3.4 Gpc (median 1.6 Gpc). It is updated when `--snr-range` is used. |
| `snr`, `snr_det` | Optimal SNRs of the whole injected signal, not matched-filter SNRs recovered from the noisy data. For light systems part of the inspiral starts before the 2 s window, so the SNR actually contained in the window can be lower (down to about 0.89 × `snr`; the median ratio is 1.00). |

## Loading

```python
import h5py, numpy as np, torch

with h5py.File("eucaif_whitened.h5", "r") as f:
    x = torch.from_numpy(f["x"][:1000])                          # (1000, 2, 4096)
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
| `--chunk-per-class` | 1000 | examples per class generated at once; peak RAM is a few GB at 1000 |
| `--store-dtype` | float32 | `float64` doubles the size and also whitens in float64 |
| `--snr-range MIN MAX` | off | rescale to a network optimal SNR uniform in [MIN, MAX]; off keeps physical distances |
| `--seed` | 150914 | reproducibility; the same seed gives the same file (up to floating-point differences between machines) |

The window is set by the data config registered in the script (`sample_length_in_s
= 2`, `padding_length_in_s = 4`). Before writing, the script runs a quick check
and saves a plot (`whitened_examples.png`). It stops with an error if the data
config is inconsistent or if any merger falls outside the window.

**Storage** (2 detectors × 4096 samples):

| Examples (both classes) | float32 | float64 |
|---|---|---|
| 10,000 | 0.33 GB | 0.66 GB |
| 76,000 | 2.5 GB | 5.0 GB |
| 150,000 | 4.9 GB | 9.8 GB |

## Relation to the ggwd setup

| | ggwd INI | This dataset |
|---|---|---|
| Masses, sky, inclination, phase, polarisation | same priors | same priors |
| `chirp_distance` | uniform in volume, 130–350 Mpc | uniform in volume, **65–175 Mpc** (half, for an easier set: SNR × 2) |
| Spins | zero | effectively zero (magnitude < 1e-6: sage's prior rejects a zero-width range) |
| Approximant | IMRPhenomXPHM, 22 mode only | IMRPhenomPv2; at zero spin the match to XPHM-22 is 0.992–0.999 |
| Distance | `chirp_distance × (Mc / 1.2188)^(5/6)` | `chirp_distance × (Mc / 1.2)^(5/6)`: distances 1.3 % larger |
| Window | 2 s, merger at 1 s (H1 arrival time) | 2 s, geocentric `tc` uniform in [0.5, 1.5] s |
| Noise | real O3 data from HDF files (synthetic aLIGO design if no event time) | synthetic aLIGO design |
| PSD for whitening | estimated from 16 s of the data itself | exact ASD that coloured the noise |
| Whitening filter / highpass | 4 s; 20 Hz FIR highpass afterwards | 4 s; 15 Hz highpass inside the filter |

## Caveats

- **SNR distribution.** Distances are physical, so SNRs follow from the
  prior. With `chirp_distance` in 65–175 Mpc and aLIGO design sensitivity,
  the network optimal SNR has median about 10 (10–90 % range: 4–19). About
  84 % of signals have SNR ≥ 5, 63 % have SNR ≥ 8 and 37 % have SNR ≥ 12. The
  roughly 16 % below SNR 5 are practically indistinguishable from noise. Use
  `metadata/snr` to weight, filter or evaluate by SNR bin. The ggwd /
  ML-challenge range (130–350 Mpc) halves every SNR (median about 5, 19 % at
  SNR ≥ 8).
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
- **Shuffling.** Examples are shuffled within each generation chunk
  (`--chunk-per-class` per class), not across the whole file. Shuffle again
  when training.

## Possible future updates: SNR control

The difficulty is currently set only through the distance prior; since SNR
scales as 1 / distance, scaling the `chirp_distance` range by a factor k
scales every SNR by 1 / k. For finer control, the generator already
supports explicit SNR control, which could be made part of the dataset
definition:

- **`--snr-range MIN MAX`** (already implemented): rescale every signal to a
  network optimal SNR drawn uniformly from the range, with `distance` updated
  to `distance / scale` so amplitude and distance stay consistent (sage's
  `OptimalSNRRescaler` convention). Distances then no longer follow the prior.
- **Other SNR distributions**, such as sage's `HalfNorm` target-SNR sampler or
  a power law in SNR, to set the balance of easy and hard examples.
- **SNR-binned or curriculum datasets**: several files with decreasing
  distance ranges or SNR ranges, for training from easy to hard and
  evaluating sensitivity per SNR bin.
- **In-window SNR** as an extra metadata field, since for light systems part
  of the inspiral lies before the 2 s window.

