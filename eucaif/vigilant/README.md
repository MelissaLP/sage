# Vigilant: whitened Virgo glitches

Whitened Virgo (V1) strain around the O3b glitches of `O3b_vigilant_train_set.csv`, made by
`save_glitches` in `virgo_data_prep.ipynb`.

## Data format

One HDF5 file, `virgo_glitches.h5`, with one row per glitch:

| Path | Shape, type | Content |
|---|---|---|
| `whitened` | (n, 131072) float32 | 32 s of whitened strain at 4096 Hz; the glitch is at the centre (sample 65536, within half a sample) |
| `metadata/GPStime` | (n,) float64 | glitch GPS time in s, rounded to 1 ms (the time the window is centred on) |
| `metadata/snr`, `metadata/confidence` | (n,) | from the CSV |
| `metadata/label`, `metadata/new_label` | (n,) string | from the CSV |

File attributes: `sample_rate`, `keep` (32 s), `total` (64 s), `highpass`, `fduration`, `gps_decimals`,
`channel`, `frametype`, `whitening`, `sage_commit` (SAGE version used).

```python
import h5py
with h5py.File("virgo_glitches.h5") as h:
    x = h["whitened"][:]                       # (n, 131072)
    gps = h["metadata/GPStime"][:]
    labels = h["metadata/label"].asstr()[:]    # strings: .asstr()
```

## Whitening

For each glitch:

1. **Data**: 64 s of `V1:Hrec_hoft_16384Hz` (frame type `HoftOnline`) centred on `GPStime`.
2. **Resampling** to 4096 Hz: `sage.dsp.decimate`, a half-band Kaiser FIR applied twice (16384 → 8192 → 4096 Hz;
   at least 92 dB suppression above 2048 Hz).
3. **ASD** from the same 64 s: `sage.dsp.welch.TorchWelch`, 4 s Hann segments, 2 s overlap, median average.
4. **Whitening**: `sage.dsp.whiten.FIRWhitening`, the same as gwpy's `TimeSeries.whiten` (gwpy >= 4): a 2 s FIR
   inverse-spectrum-truncation filter from the ASD, high-pass at 10 Hz. Computed in float64 (needed for real strain),
   stored as float32. Whitened Gaussian noise has unit variance. The filter corrupts the first and last 1 s, which are dropped.
5. **Crop**: the 32 s centred on `GPStime` (derivation: `eucaif/notes/centred_crop.pdf`).

Given the same 4096 Hz data, steps 3 and 4 match gwpy to ~1e-11. The only difference from gwpy's pipeline
is step 2: gwpy's `resample` lets through more aliasing (~60 dB suppression above 2048 Hz), which changes Q-transform
images by up to ~1.5% of their peak.
