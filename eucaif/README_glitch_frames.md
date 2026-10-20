# Glitch frames: a Q-domain DeepExtractor dataset with SAGE

Goal: train a network that extracts glitches from whitened strain in the time-frequency domain of
SAGE's **invertible Q-transform** (`QTransform.tiles` / `QTransform.inverse`), on frames full of
analytic glitches, in the spirit of DeepExtractor's training data (many glitches of random types
per frame). Unlike DeepExtractor's STFT (64-sample window, ~256 Hz wide below 100 Hz), the
Q-transform resolves low-frequency glitches such as scattered-light arches.

Run everything from the SAGE repo root with `PYTHONPATH=.`.
First steps with SAGE (Gaussian noise, whitening, Q-transform in ~20 lines): `eucaif/example_noise_qtransform.py`.

## Files

| File | What it is |
|---|---|
| `eucaif/sage_analytic_glitches_dense.yaml` | **The config**: one detector (L1; list more under `detectors` and `asd` if needed), glitch classes (Blip, Tomte, sine-Gaussian, ringdown, white-noise burst, Gaussian blip, scattered-light arch), 5-30 glitches per window, SNR range, and the `dataset` block (frame, padding, whitening, ASD) |
| `eucaif/make_glitch_dataset.py` | Generates frames and writes them to HDF5 (fixed validation / test sets) |
| `eucaif/glitch_frames.py` | `GlitchFramesOnTheFly` (training) and `StoredGlitchFrames` (validation / test): same format |
| `sage/data/noise/glitch_population.py` | Reads the config, draws and injects the glitches |
| `sage/data/noise/analytic_glitches.py`, `scattered_light.py` | The glitch shapes (sources and citations in the docstrings and in `sage/data/noise/README.md`) |
| `sage/dsp/qtransform.py` | gwpy-exact Q-transform, plus `tiles` / `inverse` (exact, differentiable) |

## The frame and its padding

```
generated (8 s):  [ 1 s dropped by whitening | 1 s pad | 4 s frame | 1 s pad | 1 s dropped by whitening ]
stored    (6 s):                             [ 1 s pad | 4 s frame | 1 s pad ]
```

- `FIRWhitening` discards `fduration / 2` (1 s) at each end, so the noise is generated longer.
- The Q-transform is circular (FFT-based): the end of the window leaks into the start over about
  one tile duration, Q' / (2 f) with Q' = Q / sqrt(11): 0.34 s at 20 Hz and 0.7 s at 10 Hz for
  Q = 45. Compute the tiles on the whole stored window and **use only the central 4 s** (loss,
  metrics). The 1 s padding covers Q = 45 down to ~10 Hz.
- Glitches can sit anywhere in the stored window (also partly in the padding), never where the
  whitening drops data. `glitches_per_segment` counts them over the stored 6 s.

Each frame stores the whitened strain `x` and the whitened glitches alone `glitch` (the target);
the whitened noise is `x - glitch` (the whitening is linear, exact to ~3e-7 in float32).

## Step 1: fixed validation and test sets

```bash
python3 eucaif/make_glitch_dataset.py eucaif/sage_analytic_glitches_dense.yaml \
    --n-frames 10000 --seed 1 --out val.h5 --fig val_check.png
python3 eucaif/make_glitch_dataset.py eucaif/sage_analytic_glitches_dense.yaml \
    --n-frames 10000 --seed 2 --out test.h5
```

About 2 GB each (float32, one detector), ~280 frames/s per CPU process. The script first prints checks on a
small chunk: whitened noise std ~1, glitches per window, SNR range, class counts. Commit the SAGE
checkout first: the file stores the commit, and warns if there are uncommitted changes.

HDF5 layout: `x`, `glitch` (n, D, 24576) with D = 1 here, `t` (time from the frame start, negative
in the left padding), `n_glitches` (n, D), `glitches/*` (one row per glitch: frame, detector, class, time,
snr and every generator parameter), attributes (classes, detectors, `frame_slice`, the full config
text, seed, command, SAGE commit).

## Step 2: train on the fly

Fresh frames every epoch from the same generator and config; batch *i* of epoch *e* always has
the same seed, whatever the number of workers. Use a training seed different from the stored sets.

```python
import torch
from torch.utils.data import DataLoader
from eucaif.glitch_frames import GlitchFramesOnTheFly, StoredGlitchFrames, frame_columns
from sage.dsp.qtransform import QTransform

train = GlitchFramesOnTheFly("eucaif/sage_analytic_glitches_dense.yaml", batch_size=32,
                             batches_per_epoch=1000, seed=0)
loader = DataLoader(train, batch_size=None, num_workers=8)      # items are whole batches
val = DataLoader(StoredGlitchFrames("val.h5"), batch_size=32, num_workers=4)

Q = 45.0                                                         # or several planes as channels
qt = QTransform(4096.0, 24576, qrange=(Q, Q), frange=(10, 1000), norm=False).cuda()
frame = slice(*train.frame_slice)                               # samples of the central 4 s
cols = frame_columns(qt, Q, 1.0, 4.0)                           # tile columns of the central 4 s

for epoch in range(n_epochs):
    train.set_epoch(epoch)
    for batch in loader:
        tiles = qt.tiles(batch["x"].cuda(), Q)                  # (B, 1, rows, cols) complex
        pred_tiles = model(tiles)                               # e.g. the glitch tiles
        pred = qt.inverse(pred_tiles, Q)                        # back to the time domain
        loss = ((pred - batch["glitch"].cuda())[..., frame] ** 2).mean()
```

- `QTransform(..., differentiable=True)` if the loss goes through `inverse` (gradients flow).
- `inverse` ignores frequency bins covered only by far window tails (`rcond=1e-3`): without it,
  modified tiles (a mask, a network output) are amplified up to ~1e4 there.
- Tiles are ~50x larger than the time series (Q = 45: ~200 rows x 4096 columns for 6 s), which
  is why we store time series and compute tiles on the GPU.
- One Q is a compromise: blips prefer Q ~ 5, arches Q ~ 45. Several planes as input channels
  (`qt.tiles(x, q)` for each q, each with its own shape) is worth trying.

Metrics on the central frame (whitened data, unit-variance noise): residual SNR of a glitch
estimate = `sqrt(sum((pred - glitch)**2))` (0 is perfect, the glitch's own SNR is no
subtraction), match = `<pred, glitch> / (|pred| |glitch|)`.

## Step 3: check what we found

Each item reproduces a result discussed with Melissa; the expected numbers are what we got.

1. **SAGE's Q-transform is gwpy's**, and the inverse is exact:
   `pytest tests/test_dsp_qtransform.py` (also round trip ~1e-16, `|tiles|^2` = gwpy's tile
   energies, masked tiles not amplified).
2. **Speed** (GV100, 4 s at 4096 Hz, batch 64): SAGE 0.42 ms/series, ~440x gwpy, ~5x ml4gw (whose
   images differ from gwpy by ~0.9 of the peak). Reproduce on a GPU node:
   `python3 eucaif/benchmark_qtransform.py --devices cuda --batches 64 --repeats 20 --duration 4`.
   For the QTAM row: `pip install --no-deps qtam` (`--no-deps` keeps your torch; qtam 0.3.1's
   own import is broken, the script works around it).
3. **Scattered light is invisible in DeepExtractor's STFT** at SNR 20 and clear in a Q-scan:
   `python3 eucaif/compare_arch_tf.py` (peak 15 vs noise peak 11 for the STFT; ~30 vs 9 for QTAM
   and SAGE at Q = 45).
4. **Best possible extraction with a mask** (oracle masks, SNR 20, 5 seeds):
   `python3 eucaif/compare_glitch_extraction.py --outdir figures/` (needs gengli for the blip, or
   `--blip-file`). Residual SNR: blip STFT 4.4, QTAM 3.4, SAGE 3.5; arch STFT 14.9, QTAM 7.4,
   SAGE 7.5. The Q-transform halves the arch residual.
5. **What limits it** (arch): `python3 eucaif/study_arch_residual.py`. Most of the residual is
   noise kept inside the mask (7.2), not glitch missed (2.1); soft masks or several Qs do not help;
   fitting the arch model gives 1.9, but only with its parameters to ~0.01 Hz and ~1 ms
   (0.2 Hz off: residual 16.8).
6. **The pretrained DeepExtractor** on the same data:
   `python3 eucaif/test_deepextractor_arch.py --deepextractor-src ../deepextractor/src
   --checkpoint <real-noise checkpoint> --blip-file <blips.npy>`. Use the **real-noise
   checkpoint with `scaler.pkl`**, input times sqrt(fs / 2) = 45.25 (the bilby checkpoint treats
   white noise as glitch). The `.pth.tar` files in the deepextractor clone are Git LFS pointers:
   `git lfs pull`, or download from Hugging Face `tomdooney/deepextractor` and check the sha256.
   Expected residual: arch 15.2 (no better than the STFT oracle), blip 7.2.
7. **Edge effects**: with `--fig`, `make_glitch_dataset.py` plots a frame; the wrap-around shows
   as bright low-frequency bands at the outer edges, inside the padding. Try `q_padding: 0.5`.
8. **Low-frequency rows need enough bins**: at Q = 45 a row at f has ~2 f T / Q' frequency bins
   (T the window length). With 2 s windows, rows below ~20 Hz have 1-3 bins and turn into
   horizontal stripes (seen with QTAM); 4-8 s windows avoid it.

## Notes

- Python >= 3.10 for parts of the SAGE test suite; NumPy 1.x works (a seed bug in
  `sample_synthetic_noise` was fixed for it).
- Seeds: stored sets use `--seed` (1, 2, ...); on-the-fly batches derive theirs from
  `(seed, epoch, batch)`. Never reuse the same seed for two stored sets.
- The SNR of each glitch is its optimal SNR against the colouring ASD; with up to 30 glitches per
  window, frames are crowded on purpose (DeepExtractor used 1-30 per 2 s, SNR uniform 1-250; here
  5-100 so quiet glitches are not all buried).
