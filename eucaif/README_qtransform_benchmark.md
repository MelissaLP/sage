# Q-transform benchmark: SAGE, ml4gw, QTAM, gwpy and DeepExtractor's STFT

`eucaif/benchmark_qtransform.py` times the time-frequency transforms we use or compare against,
on batches of whitened strain. The libraries do different amounts of work by default, so the
script compares them **one task at a time** (`--task`): every library does the same job with
the same settings (duration, sample rate, `qrange`, `frange`, mismatch, output grid,
normalisation, float32, batch size, device).

| Library | What it is |
|---|---|
| **SAGE** `sage.dsp.qtransform.QTransform` | gwpy's Q-transform in torch, batched, differentiable: `forward` (gwpy-exact image), `tiles` / `inverse` (invertible complex tiles), `best_q` |
| **ml4gw** `ml4gw.transforms` | `QScan` / `SingleQTransform`: fast Q-scan images, energies only, one Q per batch, not invertible |
| **QTAM** `qtam` | invertible Q-transform with its own frequency rows and windows; no median normalisation |
| **gwpy** | the reference (`q_transform`, `q_scan`, `QPlane`): numpy, CPU only, one series at a time |
| **DeepExtractor's STFT** | `torch.stft` / `torch.istft` (n_fft 512, Hann window 64, hop 32) |

## The five tasks

| `--task` | Job | gwpy | ml4gw | QTAM | SAGE |
|---|---|---|---|---|---|
| `scan` (default) | Q-scan image: every Q-plane, median-normalised energies, the loudest plane, interpolated onto one image | `q_transform` | `QScan` (one Q per batch) | `QScanMulti` (no median) | `forward()` |
| `singleq` | the same image at one fixed Q (`--q`) | `q_transform(qrange=(q, q))` | `SingleQTransform` | `SingleQTransform`, energy | `forward()` with `qrange=(q, q)` |
| `tiles` | raw tiles at one Q: no normalisation, no interpolation | `QPlane.transform` | `compute_qtiles` (energies) | complex tiles, full or `num_time` grid | `tiles()` (complex) |
| `roundtrip` | tiles at one Q and back to the time domain | – | – | complex tiles + `invert_qtransform` | `tiles()` + `inverse()` |
| `bestq` | only the choice of the loudest Q-plane, no image | `q_scan` | all planes + `get_max_energy` | – | `best_q()` |

The STFT appears in `scan` (DeepExtractor's magnitude + phase) and `roundtrip` (STFT + iSTFT);
`--task all` runs the five tasks. Each method also gets a check, computed on the device being
timed. Which error matters depends on the task:

- `scan`, `singleq`: largest difference from gwpy's image, relative to its peak ("gwpy err").
- `tiles`: the same for the raw tile energies, at each row's own times ("gwpy err"). SAGE's
  `|tiles|^2` and ml4gw's `compute_qtiles` are gwpy's tile energies; QTAM's tiles are not.
- `bestq`: fraction of series whose Q matches gwpy's ("Q = gwpy").
- `roundtrip`: relative reconstruction error within [2 f_low, 0.8 f_high] ("recon. err"); gwpy
  has no inverse.

QTAM and the STFT are different transforms, so they have no gwpy check except in `roundtrip`.

## Why the costs differ

- **Row lengths.** In gwpy's tiling each frequency row has its own number of time tiles (few at
  low frequency, many at high frequency). gwpy, ml4gw and SAGE's `forward` compute each row at
  that natural length. SAGE's `tiles` and QTAM put every row on one shared time grid, which a
  network needs and which keeps the transform exactly invertible. That costs 3-7x more FFT
  points for SAGE (padding to the plane's longest row) and up to about 14x for QTAM at full
  resolution (padding to the data length).
- **Median normalisation.** gwpy divides each row by its median and picks the loudest plane from
  those normalised energies. An exact per-row median on every plane is the largest single cost of
  `forward()` and `best_q` (70-75% of `forward()` at 2 kHz). ml4gw does the same with
  `torch.quantile`; QTAM does not normalise.
- **Interpolation.** SAGE's `forward` reproduces gwpy exactly (a cubic spline per row, then
  interpolation onto the output frequencies). ml4gw resizes with bicubic `F.interpolate`, one fast
  kernel, which is why it does not match gwpy (check column about 0.9). QTAM keeps its own rows.
- **One Q per batch.** ml4gw picks the plane with the loudest tile in the whole batch, so most
  series in a mixed batch get a Q other than gwpy's (50-75% agreement here). SAGE picks per series.
- **Loops in Python.** ml4gw and QTAM loop over frequency rows (one small FFT each); SAGE does one
  gather and one batched FFT per plane. On CPU the loop costs little; on GPU each row is a few
  kernel launches.
- **The inverse.** SAGE inverts directly from its grid, reading only the windowed bins (5-11% of
  the grid). QTAM with a reduced `num_time` first upsamples every row back to full resolution.

So `forward()` costing more than `tiles` + `inverse` is expected: it scans every plane with
medians and interpolates, while the round trip touches one plane with neither.

## Results

Apple M5 (virtual machine, 4 CPU cores, 8 GB), torch 2.10, float32, best of 3 runs. Milliseconds
**per series** at the largest batch (256 for 2 s, 64 for 8 s), CPU / MPS (the Mac's GPU). Raw outputs are
reproducible with the commands below.

**2 s at 4096 Hz, 20-1000 Hz, image 257 x 257 (DeepExtractor's setup), Q = 11.31 for the fixed-Q tasks**

| Task | SAGE | ml4gw | QTAM | gwpy (CPU) | STFT | Check (CPU) |
|---|---|---|---|---|---|---|
| `scan` | 3.4 / 1.5 | 2.0 / 2.5 (spline: 4.5 / fails) | 2.2 / 2.1 | 32 | 0.29 / 0.09 | gwpy err: SAGE 1.6e-6, ml4gw 0.93 |
| `singleq` | 1.5 / 0.5 | 0.57 / 0.34 (spline: 11.6 / fails) | 0.29 / 0.13 | 7.6 | – | gwpy err: SAGE 6.0e-7, ml4gw 0.98 |
| `tiles` | 0.69 / 0.09 | 0.15 / 0.15 | 1.7 / 0.49 (full), 0.41 / 0.21 (`num_time`) | 6.0 | – | gwpy err: SAGE 4.3e-7, ml4gw 3.5e-7 |
| `roundtrip` | **1.4 / 0.2** | – | 3.5 / 1.3 (full), 3.5 / 2.3 (`num_time`) | – | 0.65 / 0.35 | recon. err: all 1e-7 to 5e-7 |
| `bestq` | 1.4 / 0.7 | 1.8 / 2.5 | – | 34 | – | Q = gwpy: SAGE 100%, ml4gw 75% |

**8 s at 8192 Hz, 10-2000 Hz, image 257 x 2049, Q = 11.31 for the fixed-Q tasks**

| Task | SAGE | ml4gw | QTAM | gwpy (CPU) | STFT | Check (CPU) |
|---|---|---|---|---|---|---|
| `scan` | 37 / 9.0 | 19 / 14 (spline: killed, out of memory) | 23 / 10 | 80 | 2.3 / 0.7 | gwpy err: SAGE 7.9e-7, ml4gw 0.92 |
| `singleq` | 17 / 3.3 | 5.5 / 2.1 | 3.2 / 1.3 | 34 | – | gwpy err: SAGE 5.7e-7, ml4gw 0.85 |
| `tiles` | 8.7 / 1.2 | 1.5 / 0.9 | 24 / 5.4 (full), 6.7 / 2.2 (`num_time`) | 12 | – | gwpy err: SAGE 4.0e-7, ml4gw 4.0e-7 |
| `roundtrip` | **16 / 2.7** | – | 53 / out of memory (full), 65 / out of memory (`num_time`) | – | 5.1 / 2.3* | recon. err: all 1e-7 to 6e-7 |
| `bestq` | 15 / 4.2 | 18 / 14 | – | 51 | – | Q = gwpy: SAGE 100%, ml4gw 50% |

\* The STFT ran out of memory on MPS after QTAM's failures in the same process (MPS does not
return that memory); 2.3 ms is from a run without QTAM (`--skip QTAM`). On MPS the checks are
about 1e-6 to 5e-6 (float32 on the GPU), with the same Q agreement.

What this shows (the answer depends on the task):

- **Only SAGE and ml4gw's raw tiles reproduce gwpy's numbers**, and only SAGE does so for the
  images and the Q choice (images to about 1e-6, the same Q for every series). SAGE is 1.3-24x
  faster than gwpy's loop on CPU (most for small inputs and for `bestq`).
- **For images, the gwpy-exact spline and medians cost something.** On CPU, ml4gw's bicubic image is
  1.7-3x faster than SAGE's `forward` and QTAM's single-Q image (its own rows, no median) is
  faster still. On MPS the gap closes: SAGE's full Q-scan is the fastest of the three in both setups.
- **For raw tiles, ml4gw is fastest on CPU** because it keeps each row at its natural length (and
  energies only); SAGE pays for its shared, invertible grid. On MPS the two are close (SAGE ahead
  at 2 s, ml4gw at 2 kHz).
- **For the round trip, SAGE is the fastest Q-transform**: 2.4-4x faster than QTAM on CPU and 6x
  on MPS at 2 s, where it is also faster than the STFT (0.2 vs 0.35 ms per series). QTAM ran out
  of memory on MPS at 2 kHz.
- **Picking Q per series costs about the same as ml4gw's per-batch choice on CPU**, and is 3-4x
  faster on MPS, while matching gwpy.
- **ml4gw has a fixed cost of about 0.55-0.9 s per call on MPS** for `QScan` (even at batch 1),
  probably from synchronising the GPU when comparing the planes.

## Reproduce, with figures

```bash
# from the SAGE repo root; ml4gw needs Python >= 3.11; QTAM: pip install --no-deps qtam
python eucaif/benchmark_qtransform.py --task all --outdir results/                      # DeepExtractor setup
python eucaif/benchmark_qtransform.py --task all --duration 8 --sample-rate 8192 --frange 10 2000 \
    --batches 1 16 64 --skip spline --outdir results_2khz/                             # 2 kHz setup
python eucaif/benchmark_qtransform.py --plot-only results/qtransform_benchmark.json    # redraw only
```

`--outdir` saves every timing and check to `qtransform_benchmark.json` and draws **one figure per
device**, `qtransform_benchmark_<device>.png` and `.pdf`: for each task, the time per series at
the largest batch, with each method's check next to its bar. gwpy (CPU only) is drawn in every
figure as the reference. `--devices` defaults to the CPU plus whichever GPU torch finds:
`cuda` on an NVIDIA machine, `mps` on an Apple-silicon Mac (Metal Performance Shaders, Apple's
GPU backend for torch). Single tasks run with `--task scan singleq ...`.

**On a cluster (e.g. Nikhef).** Run the CPU job on a CPU node and the GPU job on a GPU node, each
into its own folder, since the two machines differ:

```bash
python eucaif/benchmark_qtransform.py --task all --devices cpu  --outdir results_cpu/
python eucaif/benchmark_qtransform.py --task all --devices cuda --outdir results_gpu/
```

The CPU timings depend on the number of threads torch uses (`OMP_NUM_THREADS`, or the cores
the job was given); the figure's footnote records it.

`--skip` leaves out methods by name (`spline`, `QTAM`, `gwpy`, ...). ml4gw's spline interpolation
builds large matrices and was killed for lack of memory at 8 s here; it also fails on MPS
(`torch.linalg.solve`: singular matrix). Methods that fail or run out of memory are reported as
such and the run continues. The script works around an import bug in QTAM 0.3.1 (it imports a
bundled helper as a top-level module), so `pip install --no-deps qtam` is enough.

## Caveats

- The timings above come from a 4-core, 8 GB virtual Mac. CUDA GPUs and larger machines will rank
  the Python row loops (ml4gw, QTAM) and the out-of-memory cases differently; run the script there.
- The data are white noise with sine-Gaussian glitches. Run time does not depend on the data,
  except through which Q-plane gets chosen.
- Speed only. How well each representation supports glitch removal is measured separately, with
  an oracle mask, in `residual_snr_by_class.py` and `compare_glitch_extraction.py`.
