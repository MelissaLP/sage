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

The jobs run in a container (`universe = container`, as the anomalydetection CIT pipeline), at CIT where the frames
are. The image (`sage_cpu.def`: Python 3.11, CPU torch, gwpy, h5py, a frame reader) holds no SAGE: each job gets
SAGE's `sage/`, `prep.py` and the CSV in its sandbox, so code changes need no rebuild. The old `anomaly` image is
Python 3.9, which SAGE does not run on.

Once, build the image and put it on OSDF (on a CIT login node, with a token as in anomalydetection's
`running_on_condor.md`):

```bash
apptainer build sage_cpu.sif eucaif/vigilant/sage_cpu.def
htgettoken -v -a vault.ligo.org -i igwn --scopes "read:/staging write:/staging"
pelican object put sage_cpu.sif osdf:///igwn/cit/staging/$USER/sage_cpu.sif
```

Then, from the SAGE repo root, in a shell with h5py and pandas (e.g. `conda activate igwn`; the merge runs there):

```bash
CSV=eucaif/vigilant/O3b_vigilant_train_set.csv; OUT=/path/on/shared/disk/vigilant
PYTHONPATH=. python -m eucaif.vigilant.prep dag $CSV $OUT --chunk 1500   # --image osdf:///.../sage_cpu.sif if not melissa.lopez's
condor_submit_dag $OUT/vigilant.dag
```

- One container job per `--chunk` glitches (`prep.sub`) sends its two files back to `OUT/parts/`. When all are
  done, `merge.sub` (on the submit machine) joins them, in CSV order, into `OUT/virgo_glitches_{median,welch}.h5`
  and writes `OUT/virgo_glitches.tar` (both files and this README, uncompressed: whitened noise does not compress).
- Accounting group: `ligo.dev.o4.detchar.glitch_classification.lib` (`--accounting-group` to change it).
- Pick `--chunk` for ~0.5 h per job: 1800 s / (seconds per sample), printed at the end of each part's log
  (`OUT/logs/part_*.out`). Try a small CSV first (e.g. `head -21 $CSV > test.csv`, `--chunk 10`: 2 jobs).
- Each part is retried twice. After a failure, run `prep.py dag` again: it leaves out the parts already in
  `OUT/parts/`; then `condor_submit_dag -force $OUT/vigilant.dag`.
- Disk: parts, merged files and tar each take ~57 GB for 12k glitches; the parts can be deleted after the merge.
