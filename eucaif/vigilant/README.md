# Vigilant: whitened Virgo glitches

Whitened Virgo (V1) strain around the O3b glitches of a glitch CSV: `virgo_o3b_no_NOTA_no_rareclasses_no_Whis_from_v4.csv`
(12,186 glitches, from virgo-glitch-classification's `data/csvs/`, copied next to this README; not committed), made by `prep.py`
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
| `metadata/<column>` | (n,) | every other column of the CSV (`label`, `snr`, `confidence`, ...; text as strings) |

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

The jobs run in a container (`universe = container`, as in the anomalydetection CIT pipeline), on any IGWN node
with the IGWN private data (OSDF, in CVMFS): gwpy's `TimeSeries.get` finds h(t) with `GWDATAFIND_SERVER=datafind.igwn.org`
and reads it with the job's SciToken (`read:/virgo gwdatafind.read`). They see nothing of your home directory: HTCondor transfers SAGE's `sage/`, `prep.py` and the CSV into
each job's sandbox, and transfers the job's two files back. The image (`sage_cpu.def`: Python 3.11, CPU torch,
gwpy 4, h5py, a frame reader) holds no SAGE, so code changes need no rebuild. The old `anomaly` image is Python 3.9,
which SAGE does not run on.

```
CSV ── prep.py dag ──> OUT/vigilant.dag
                          │
       one container job per --chunk glitches (prep.sub, h(t) from CVMFS, retried twice)
         in:  sage/, prep.py, the CSV (transfer_input_files)
         out: OUT/parts/virgo_glitches_partNNN_{median,welch}.h5 (transfer_output_files)
                          │
       merge: a node that submits nothing; its PRE script (OUT/merge.sh) runs prep.py merge on the
       submit machine, with the Python of `prep.py dag`, once every part has succeeded
                          │
OUT/virgo_glitches_median.h5, OUT/virgo_glitches_welch.h5, OUT/virgo_glitches.tar (both + this README)
```

### 1. Set up (once)

On a CIT login node (e.g. `ldas-osg`):

```bash
# SAGE on the eucaif branch
git -C ~/sage checkout eucaif && git -C ~/sage pull

# the glitch list, next to this README. In virgo-glitch-classification the CSVs are in Git LFS: pull them first,
# or the copy is a 3-line pointer ("version https://git-lfs...") instead of the glitches
git -C ~/virgo-glitch-classification lfs pull
cp ~/virgo-glitch-classification/data/csvs/virgo_o3b_no_NOTA_no_rareclasses_no_Whis_from_v4.csv ~/sage/eucaif/vigilant/
head -3 ~/sage/eucaif/vigilant/virgo_o3b_no_NOTA_no_rareclasses_no_Whis_from_v4.csv   # id,GPStime,... and two glitches

# a Python with h5py and pandas, for prep.py dag/parts and the merge (the jobs use the image instead)
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn

# the image: build it and put it on OSDF (~1 GB). Rebuild only if sage_cpu.def changes
cd ~/sage
apptainer build sage_cpu.sif eucaif/vigilant/sage_cpu.def      # ends with "ok 2.10.0 ..." from its %test
htgettoken -v -a vault.ligo.org -i igwn --scopes "read:/staging write:/staging"
pelican object put sage_cpu.sif osdf:///igwn/cit/staging/melissa.lopez/sage_cpu.sif
pelican object ls osdf:///igwn/cit/staging/melissa.lopez/      # check it is there
```

`osdf:///igwn/cit/staging/melissa.lopez/sage_cpu.sif` is the default image; someone else passes theirs with
`--image`.

### 2. Test with 20 glitches

In the shell of step 1. `CSV` is the glitch list; `TEST` (and `OUT` in step 3) are output folders, outside the repo
so `git status` stays clean:

```bash
cd ~/sage
CSV=/home/melissa.lopez/sage/eucaif/vigilant/virgo_o3b_no_NOTA_no_rareclasses_no_Whis_from_v4.csv
TEST=/home/melissa.lopez/vigilant_test                            # ~200 MB

head -21 $CSV > $TEST.csv                                         # header + 20 glitches: /home/melissa.lopez/vigilant_test.csv
python eucaif/vigilant/prep.py dag $TEST.csv $TEST --chunk 10     # 2 jobs, then merge
condor_submit_dag $TEST/vigilant.dag
```

When the DAG is done (`condor_q` empty):

```bash
tail -3 $TEST/logs/part_0.out        # "time per sample (...): mean X s": X is what step 3 needs
cat $TEST/logs/part_*.err            # empty, or warnings only
cat $TEST/logs/merge.out             # "... 20 glitches from 2 parts, 0 skipped", "-> .../virgo_glitches.tar"
python -c "import h5py; h = h5py.File('$TEST/virgo_glitches_median.h5'); print(h['whitened'].shape, list(h['metadata']))"
```

`(20, 524288)` and the CSV's columns (`GPStime`, `label`, `split`, ...) mean it works. If the parts fail reading
the data (`RuntimeError`/`FileNotFoundError` from `TimeSeries.get` in `part_*.out`, every glitch "skipped"), the
job cannot reach h(t): stop there and check it in a debug job ("Run prep.py by hand"):

```bash
python -c "from gwdatafind import find_urls; print(find_urls('V', 'HoftOnline', 1264551656, 1264551720))"
ls /cvmfs/virgo.storage.igwn.org/ | head        # empty or an error: no token or no CVMFS on that node
```

`find_urls` is what `TimeSeries.get` asks; it should give files the job can open (expected under
`/cvmfs/virgo.storage.igwn.org/`).

### 3. Run all glitches

Pick `--chunk` for ~0.5 h per job: 1800 / X, with X the seconds per sample of step 2 (1500 if X is ~1.2 s).

```bash
OUT=/home/melissa.lopez/vigilant                                  # needs ~175 GB at the peak, see 6: check the quota
quota -s; df -h $(dirname $OUT)                                   # if /home is too small, use e.g. /scratch
python eucaif/vigilant/prep.py parts $CSV --chunk 1500            # number of jobs: 9 for 12,186 glitches
python eucaif/vigilant/prep.py dag $CSV $OUT --chunk 1500
condor_submit_dag $OUT/vigilant.dag
```

Options of `prep.py dag`: `--chunk` (glitches per job), `--image` (container), `--accounting-group` (default
`ligo.dev.o4.detchar.glitch_classification.lib`), `--tar` (default `virgo_glitches.tar`).

### 4. Follow it

```bash
condor_q -dag -nobatch                          # the jobs of the DAG
tail -f $OUT/vigilant.dag.dagman.out            # DAG progress
tail $OUT/logs/part_3.out                       # a part: "100 of 1500 done, X s per sample", every 100 glitches
ls $OUT/parts/                                  # the parts sent back so far
```

- Logs: `$OUT/logs/part_<k>.{out,err}`, `merge.out`, and the HTCondor event log `vigilant.log`.
- The merge (the last node) takes a while: ~58 GB read, ~115 GB written. `tail -f $OUT/logs/merge.out`.
- Glitches without data are skipped and listed in the files' `skipped_rows` attribute (CSV rows); the part goes on.
- A job on hold: `condor_q -hold` says why. Out of memory: `condor_qedit <id> RequestMemory=8000`; expired
  token: `htgettoken ...` as in step 1; then `condor_release <id>`.

### 5. If something fails

Each part is retried twice. If a part still fails, the DAG stops before the merge. Fix the cause, then write the
DAG again (it leaves out the parts already in `$OUT/parts/`; with all of them there, only the merge is left) and
submit it:

```bash
python eucaif/vigilant/prep.py dag $CSV $OUT --chunk 1500         # same --chunk as before!
condor_submit_dag -force $OUT/vigilant.dag
```

Keep the same `--chunk`: the parts are numbered by it. To start over, delete `$OUT`.

### Run prep.py by hand

To test or debug a part without the DAG. `prep.py` finds SAGE by itself (next to it in a job, the repo otherwise).

On an execute node, exactly as a job (same image, files and token): submit a job that only sleeps, and log into it
(`condor_submit -i` is not allowed in the pool):

```bash
condor_submit eucaif/vigilant/debug.sub csv=$TEST.csv            # prints the job id, e.g. 1234567
condor_q 1234567                                                   # wait until it runs (R)
condor_ssh_to_job 1234567                                          # a shell in the sandbox: sage/, prep.py, the CSV
python prep.py run vigilant_test.csv . --chunk 10 --part 0        # -> ./parts/virgo_glitches_part000_{median,welch}.h5
exit
condor_rm 1234567                                                  # else it ends by itself after 2 h (arguments in debug.sub)
```

Results stay in the sandbox and are deleted with it. On the login node, in the image (your home is visible there):

```bash
apptainer exec sage_cpu.sif python eucaif/vigilant/prep.py run $TEST.csv /home/melissa.lopez/vigilant_hand --chunk 10 --part 0
```

The login node may read the frames differently from the execute nodes, so only the first checks what the jobs see.

### 6. The result

When the DAG is done, `$OUT/logs/merge.out` ends with the two files and the tar (if a part is missing, it lists it
and the merge node fails: see 5). The merge can also be run by hand, e.g. for another tar name:
`python eucaif/vigilant/prep.py merge $CSV $OUT --chunk 1500 --tar other.tar`.

- `$OUT/virgo_glitches_median.h5`, `$OUT/virgo_glitches_welch.h5`: ~29 GB each for 12,186 glitches (format above).
- `$OUT/virgo_glitches.tar`: both and this README, ~58 GB (uncompressed: whitened noise does not compress).
- Disk: the parts, the merged files and the tar take ~58 GB each, ~175 GB at the peak. After checking `merge.out`
  (all glitches, the expected number skipped), `rm -r $OUT/parts` frees ~58 GB; keep either the tar or the two
  `.h5` files.
