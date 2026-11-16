# Vigilant: whitened Virgo glitches

Whitened Virgo (V1) strain around the O3b glitches of `O3b_vigilant_train_set.csv`, made by `prep.py`
(`save_glitches`; on HTCondor, see below; `virgo_data_prep.ipynb` does a few glitches interactively).

## Data format

Two HDF5 files with the same glitches and layout, differing only in the PSD average used for whitening:
`virgo_glitches_median.h5` (median, as gwpy's `method="median"`) and `virgo_glitches_welch.h5` (mean, as gwpy's
`method="welch"`). One row per glitch:

| Path | Shape, type | Content |
|---|---|---|
| `whitened` | (n, 524288) float32 | 32 s of whitened strain at 16384 Hz; the glitch is at the centre (sample 262144, within half a sample) |
| `psd` | (n, 32769) float64 | the one-sided PSD (1/Hz) used to whiten each row |
| `frequencies` | (32769,) float64 | frequencies of the PSD bins (Hz, 0.25 Hz apart) |
| `metadata/row` | (n,) int64 | row of the CSV |
| `metadata/GPStime` | (n,) float64 | glitch GPS time in s, rounded to 1 ms (the time the window is centred on) |
| `metadata/snr`, `metadata/confidence` | (n,) | from the CSV |
| `metadata/label`, `metadata/new_label` | (n,) string | from the CSV |

File attributes: `sample_rate`, `keep` (32 s), `total` (64 s), `highpass`, `fduration`, `gps_decimals`,
`channel`, `frametype`, `psd_average` (`median` or `mean`), `skipped_rows` (CSV rows without data), `whitening`,
`sage_commit` (SAGE version used). About 2.4 MB per glitch and file: 28 GB per file for 12k glitches.

```python
import h5py
with h5py.File("virgo_glitches_median.h5") as h:   # or virgo_glitches_welch.h5
    x = h["whitened"][:]                       # (n, 524288)
    gps = h["metadata/GPStime"][:]
    labels = h["metadata/label"].asstr()[:]    # strings: .asstr()
```

## Q-transform images

Gravity Spy-style images of every glitch in 0.5, 1, 2 and 4 s windows, in one call (multi-window `outseg`,
idea from Guillem). The glitch is at the centre of each 32 s series, so the windows are centred there:

```python
import h5py, torch
from sage.dsp.qtransform import QTransform

with h5py.File("virgo_glitches_median.h5") as h:
    x = torch.tensor(h["whitened"][:64])       # 64 glitches, (64, 524288)
    fs = float(h.attrs["sample_rate"])
c = x.shape[-1] / fs / 2                        # the glitch: 16 s
windows = [(c - d / 2, c + d / 2) for d in (0.5, 1.0, 2.0, 4.0)]
qt = QTransform(fs, x.shape[-1], outseg=windows, tbin=256, logf=True, fbin=256, frange=(10, 1024))
img = qt(x, gps=c)                              # (64, 4, 256, 256): window, time, frequency
```

- `tbin` / `fbin` set the image size (times per window, frequencies). The windows don't need to be multiples of
  each other.
- `gps=c` picks the Q-plane from the loudest tile within ±0.25 s of the glitch (`search`, as gwpy), the same Q
  for all four windows. Each window is exactly what a single-window `QTransform(outseg=...)` gives, but the
  transform and the Q choice are computed once (about 3× faster than four calls).
- Building `qt` takes a few seconds (the tiling of 32 s); build it once and reuse it. On a GPU:
  `qt = qt.to("cuda")` and `x = x.cuda()`.
- `qt.times[w]` and `qt.frequencies` give each image's axes, in s from the start of the 32 s and in Hz.

## Whitening

For each glitch:

1. **Data**: 64 s of `V1:Hrec_hoft_16384Hz` (frame type `HoftOnline`) centred on `GPStime`, at its native 16384 Hz
   (no resampling).
2. **PSD** from the same 64 s: `sage.dsp.welch.TorchWelch`, 4 s Hann segments, 2 s overlap, median average
   (`_median` file) or mean average (`_welch` file); the ASD is its square root.
3. **Whitening**: `sage.dsp.whiten.FIRWhitening`, the same as gwpy's `TimeSeries.whiten` (gwpy >= 4): a 2 s FIR
   inverse-spectrum-truncation filter from the ASD, high-pass at 10 Hz. Computed in float64 (needed for real strain),
   stored as float32. Whitened Gaussian noise has unit variance. The filter corrupts the first and last 1 s, which are dropped.
4. **Crop**: the 32 s centred on `GPStime` (derivation: `eucaif/notes/centred_crop.pdf`).

Given the same data, steps 2 and 3 match gwpy to ~1e-11.

## HTCondor

From the SAGE repo root, in a shell with the SAGE environment active (the jobs inherit it):

```bash
CSV=eucaif/vigilant/O3b_vigilant_train_set.csv; OUT=/path/on/shared/disk/vigilant
PYTHONPATH=. python -m eucaif.vigilant.prep dag $CSV $OUT --chunk 1500   # accounting group ligo.dev.o4.detchar.glitch_classification.lib
condor_submit_dag $OUT/vigilant.dag
```

- One job per `--chunk` glitches (`prep.sub`) writes `OUT/parts/`; when all are done, `merge.sub` joins them, in CSV
  order, into `OUT/virgo_glitches_{median,welch}.h5` and writes `OUT/virgo_glitches.tar` (both files and this README,
  uncompressed: whitened noise does not compress).
- Pick `--chunk` for ~0.5 h per job: 1800 s / (seconds per sample), printed at the end of each part's log
  (`OUT/logs/part_*.out`). `prep.py parts $CSV --chunk N` prints the number of jobs.
- A part whose files exist is skipped, so a failed DAG can be resubmitted (`condor_submit_dag` uses the rescue DAG)
  without redoing finished parts. Each part is retried twice.
- Disk: parts, merged files and tar each take ~57 GB for 12k glitches; the parts can be deleted after the merge.
