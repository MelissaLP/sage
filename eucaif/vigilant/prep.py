"""
prep.py -- whitened Virgo (V1) strain around the glitches of a CSV, at the native 16384 Hz, whitened twice: with
the median- and with the mean-averaged (Welch) PSD. One HDF5 file per average (README.md has the format).

    PYTHONPATH=. python -m eucaif.vigilant.prep parts CSV --chunk 1500             # number of jobs
    PYTHONPATH=. python -m eucaif.vigilant.prep run   CSV OUT --chunk 1500 --part 0
    PYTHONPATH=. python -m eucaif.vigilant.prep merge CSV OUT --chunk 1500 --tar virgo_glitches.tar
    PYTHONPATH=. python -m eucaif.vigilant.prep dag   CSV OUT --chunk 1500 --image IMAGE              # HTCondor

run does the rows [part * chunk, (part + 1) * chunk) of the CSV into OUT/parts/ and skips a part whose files are
already there. merge joins the parts in CSV order into OUT/virgo_glitches_{median,welch}.h5 and, with --tar, puts
both and README.md in OUT/<tar> (uncompressed: whitened noise does not compress). dag writes OUT/vigilant.dag: one
container job per part not yet in OUT/parts (prep.sub), then merge on the submit machine (merge.sub); so after a
failure, run dag again and resubmit. Only run needs torch, gwpy and SAGE (imported there): merge, parts and dag need
h5py and pandas.
"""
import argparse
import os
import subprocess
import tarfile
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                                      # the SAGE repo (in a job's sandbox: not a repo)
CHANNEL, FRAMETYPE = "V1:Hrec_hoft_16384Hz", "HoftOnline"
AVERAGES = {"median": "median", "welch": "mean"}   # file tag -> TorchWelch avg_method (gwpy's method="median" / "welch")
DEFAULT_IMAGE = "osdf:///igwn/cit/staging/melissa.lopez/sage_cpu.sif"   # sage_cpu.def, uploaded as in README.md


def generate_data(start, end, highpass=10, keep=32):
    """V1 strain in [start, end] at its native rate, whitened with its own PSD, and the keep s centred on the
    middle of [start, end] kept. The PSD is estimated twice, median and mean (Welch) average, and the data
    whitened with each. Returns fs and {tag: (white, f, psd)} for tag in AVERAGES (white float32, keep * fs samples).

        fs, out = generate_data(t0 - 32, t0 + 32)     # 64 s fetched -> 32 s centred on t0
        white, f, psd = out["median"]
    """
    import torch
    from gwpy.timeseries import TimeSeries
    from sage.dsp.welch import TorchWelch
    from sage.dsp.whiten import FIRWhitening

    data = TimeSeries.get(CHANNEL, start, end, frametype=FRAMETYPE)
    fs = int(data.sample_rate.value)
    x = torch.tensor(data.value, dtype=torch.float64)

    # Whiten: filters in float64 (needed for real strain), returns float32; drops fduration / 2 = 1 s at each end
    whiten = FIRWhitening(fs, highpass=highpass, output_dtype=torch.float32)

    # Crop: the keep s centred on the middle of [start, end] (white[0] is at data.t0 + 1 s)
    t0 = (start + end) / 2
    # i0 = round[(t0 - t_d - fduration / 2 - keep / 2) * fs], t_d = data.t0: derivation and picture in
    # eucaif/notes/centred_crop.pdf
    i0 = int(round((t0 - data.t0.value - whiten.pad / fs) * fs)) - keep * fs // 2
    if i0 < 0 or i0 + keep * fs > x.numel() - 2 * whiten.pad:
        raise ValueError("keep is too long for [start, end]: fetch more or keep less")

    out = {}
    for tag, avg in AVERAGES.items():
        # PSD: Welch, 4 s Hann segments, 2 s overlap, median or mean average
        f, psd = TorchWelch(delta_t=1 / fs, seg_len=4 * fs, seg_stride=2 * fs, avg_method=avg)(x, return_freqs=True)
        white = whiten(x, psd.sqrt())
        out[tag] = white[i0:i0 + keep * fs], f, psd
    return fs, out


def save_glitches(df, out_dir, n=None, total=64, keep=32, highpass=10, name="virgo_glitches"):
    """Writes out_dir / f"{name}_{tag}.h5" for tag in AVERAGES: the first n glitches of df (all with n=None),
    whitened with the median- and with the mean-averaged (Welch) PSD. Glitches are written as they come, so memory
    stays small; files are written as .tmp and renamed at the end. Glitches without data are skipped and listed
    in the attribute skipped_rows (df index)."""
    df = df if n is None else df.head(n)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {tag: out_dir / f"{name}_{tag}.h5" for tag in AVERAGES}
    tmp = {tag: p.with_suffix(".h5.tmp") for tag, p in paths.items()}
    files = {tag: h5py.File(p, "w") for tag, p in tmp.items()}
    rows, skipped, seconds = [], [], []
    try:
        for i, row in df.iterrows():
            t0 = round(float(row['GPStime']), 3)                # GPS time to 1 ms: the window is centred on it
            tic = time.perf_counter()
            try:
                fs, out = generate_data(t0 - total / 2, t0 + total / 2, highpass=highpass, keep=keep)
            except Exception as e:                              # e.g. no data at that time: skip, keep going
                print(f"row {i} (GPS {t0}): skipped, {type(e).__name__}: {e}", flush=True)
                skipped.append(i)
                continue
            for tag, (w, f, psd) in out.items():
                h = files[tag]
                if "whitened" not in h:                         # first glitch: datasets that grow by one row each
                    h.create_dataset("whitened", (0, w.numel()), "float32", maxshape=(None, w.numel()),
                                     chunks=(1, w.numel()))     # (n, keep * fs), glitch at the centre
                    h.create_dataset("psd", (0, psd.numel()), "float64", maxshape=(None, psd.numel()),
                                     chunks=(1, psd.numel()))   # (n, 2 * fs + 1), one-sided PSD [1/Hz]; float64: a strain PSD (~1e-46) underflows float32
                    h["frequencies"] = f.numpy()                # (2 * fs + 1,) Hz, of the PSD bins (0.25 Hz apart)
                    h.attrs["sample_rate"] = fs
                k = len(rows)
                h["whitened"].resize(k + 1, axis=0); h["whitened"][k] = w.numpy()
                h["psd"].resize(k + 1, axis=0); h["psd"][k] = psd.numpy()
            rows.append(i)
            seconds.append(time.perf_counter() - tic)           # this sample: fetch + 2 x (PSD + whiten) + crop
            if len(rows) % 100 == 0:
                print(f"{len(rows)} of {len(df)} done, {np.mean(seconds):.3f} s per sample", flush=True)
        if not rows:
            raise RuntimeError("no glitch could be processed")
        meta = df.loc[rows]
        commit = sage_commit()
        for tag, avg in AVERAGES.items():
            h = files[tag]
            g = h.create_group("metadata")
            g["row"] = np.asarray(rows, dtype=np.int64)         # row of the CSV
            for col in meta.columns:                            # every CSV column but pandas' saved index
                if str(col).startswith("Unnamed:"):
                    continue
                v = meta[col].to_numpy()
                if col == 'GPStime':
                    v = np.round(v.astype(np.float64), 3)       # stored to 1 ms, as used for the centring
                name = str(col).replace("/", "_")
                if v.dtype == object:
                    g.create_dataset(name, data=v.astype(str).tolist(), dtype=h5py.string_dtype())
                else:
                    g[name] = v
            h.attrs.update(keep=keep, total=total, highpass=highpass, fduration=2.0, gps_decimals=3,
                           channel=CHANNEL, frametype=FRAMETYPE, psd_average=avg,
                           skipped_rows=np.asarray(skipped, dtype=np.int64),
                           whitening=f"SAGE TorchWelch (4 s, {avg}) -> FIRWhitening (float64), no resampling",
                           sage_commit=commit)
    finally:
        for h in files.values():
            h.close()
    for tag in AVERAGES:
        tmp[tag].rename(paths[tag])
        print(f"saved {len(rows)} of {len(df)} glitches to {paths[tag].resolve()}", flush=True)
    s = np.array(seconds)                                       # time per generated sample (skipped ones not counted)
    var = s.var(ddof=1) if s.size > 1 else 0.0
    print(f"time per sample ({s.size} samples): mean {s.mean():.3f} s, variance {var:.3g} s^2 (std {var ** 0.5:.3f} s), "
          f"min {s.min():.3f} s, max {s.max():.3f} s", flush=True)


def sage_commit():
    """The SAGE commit: SAGE_COMMIT (set by the DAG, as a job's sandbox is not a repo) or git."""
    if os.environ.get("SAGE_COMMIT"):
        return os.environ["SAGE_COMMIT"]
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:                                            # no git (a container)
        return ""


def n_parts(n_rows, chunk):
    return -(-n_rows // chunk)


def part_name(part):
    return f"virgo_glitches_part{part:03d}"


def merge(out_dir, parts, name="virgo_glitches", tar=None):
    """Joins OUT/parts/<part_name(p)>_<tag>.h5, p = 0 .. parts - 1, into OUT/<name>_<tag>.h5, in CSV order,
    copying a glitch at a time; with tar, also writes OUT/<tar> with both files and README.md."""
    out_dir = Path(out_dir)
    src = {tag: [out_dir / "parts" / f"{part_name(p)}_{tag}.h5" for p in range(parts)] for tag in AVERAGES}
    missing = [str(p) for ps in src.values() for p in ps if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} part files missing, e.g. {missing[0]}: resubmit those parts")
    finals = []
    for tag, ps in src.items():
        path = out_dir / f"{name}_{tag}.h5"
        tmp = path.with_suffix(".h5.tmp")
        with h5py.File(ps[0]) as first:
            attrs = dict(first.attrs)
            freqs = first["frequencies"][:]
            nt, nf = first["whitened"].shape[1], first["psd"].shape[1]
            cols = list(first["metadata"])
        sizes, meta, skipped, commits = [], {c: [] for c in cols}, [], set()
        for p in ps:
            with h5py.File(p) as h:
                for k in ("sample_rate", "keep", "total", "highpass", "psd_average"):
                    if h.attrs[k] != attrs[k]:
                        raise ValueError(f"{p}: {k} = {h.attrs[k]}, but {attrs[k]} in {ps[0]}")
                if not np.array_equal(h["frequencies"][:], freqs):
                    raise ValueError(f"{p}: different PSD frequencies")
                sizes.append(h["whitened"].shape[0])
                for c in cols:
                    d = h["metadata"][c]
                    meta[c].append(d.asstr()[:] if d.dtype.kind == "O" else d[:])
                skipped.append(h.attrs["skipped_rows"])
                commits.add(str(h.attrs["sage_commit"]))
        n = sum(sizes)
        with h5py.File(tmp, "w") as out:
            w = out.create_dataset("whitened", (n, nt), "float32", chunks=(1, nt))
            q = out.create_dataset("psd", (n, nf), "float64", chunks=(1, nf))
            out["frequencies"] = freqs
            k = 0
            for p, m in zip(ps, sizes):
                with h5py.File(p) as h:
                    for j in range(m):                          # a glitch at a time: memory stays small
                        w[k + j] = h["whitened"][j]
                        q[k + j] = h["psd"][j]
                k += m
            g = out.create_group("metadata")
            for c, vs in meta.items():
                v = np.concatenate(vs)
                if v.dtype.kind in "OU":
                    g.create_dataset(c, data=v.astype(str).tolist(), dtype=h5py.string_dtype())
                else:
                    g[c] = v
            attrs.update(skipped_rows=np.concatenate(skipped).astype(np.int64), parts=parts,
                         sage_commit=",".join(sorted(commits)))
            out.attrs.update(attrs)
        tmp.rename(path)
        finals.append(path)
        print(f"{path}: {n} glitches from {parts} parts, {len(attrs['skipped_rows'])} skipped", flush=True)
    if tar:
        with tarfile.open(out_dir / tar, "w") as t:
            for p in finals:
                t.add(p, arcname=p.name)
            t.add(HERE / "README.md", arcname="README.md")
        print(f"-> {out_dir / tar}", flush=True)


def write_dag(csv, out_dir, parts, chunk, accounting_group, tar, image, retries=2):
    """OUT/vigilant.dag: a container job per part not yet in OUT/parts (prep.sub, retried), then merge.sub."""
    out_dir = Path(out_dir).resolve()
    for d in ("logs", "parts"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    done = [k for k in range(parts)
            if all((out_dir / "parts" / f"{part_name(k)}_{tag}.h5").exists() for tag in AVERAGES)]
    todo = [k for k in range(parts) if k not in done]
    disk_gb = int(np.ceil(chunk * 2 * 2.4e-3)) + 2                # a part's two files, ~2.4 MB per glitch each
    common = (f'repo="{ROOT}" csv="{Path(csv).resolve()}" out="{out_dir}" chunk="{chunk}" '
              f'acct="{accounting_group}" commit="{sage_commit()}"')
    lines = []
    for k in todo:
        lines += [f"JOB part{k} {HERE / 'prep.sub'}",
                  f'VARS part{k} {common} part="{k}" image="{image}" disk="{disk_gb}"', f"RETRY part{k} {retries}"]
    lines += [f"JOB merge {HERE / 'merge.sub'}", f'VARS merge {common} tar="{tar}"']
    if todo:
        lines += ["PARENT " + " ".join(f"part{k}" for k in todo) + " CHILD merge"]
    path = out_dir / "vigilant.dag"
    path.write_text("\n".join(lines) + "\n")
    print(f"-> {path}: {len(todo)} of {parts} parts of {chunk} glitches to do ({len(done)} done), then merge\n"
          f"   condor_submit_dag {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["parts", "run", "merge", "dag"])
    p.add_argument("csv", help="glitch list with a GPStime column; all its columns are kept as metadata")
    p.add_argument("out", nargs="?", help="output directory (run, merge)")
    p.add_argument("--chunk", type=int, default=1500, help="glitches per job")
    p.add_argument("--part", type=int, default=0, help="which part to do (run): rows part * chunk ...")
    p.add_argument("--tar", help="merge: also write this tar of the merged files and README.md, in OUT "
                                 "(dag: default virgo_glitches.tar)")
    p.add_argument("--accounting-group", default="ligo.dev.o4.detchar.glitch_classification.lib",
                   help="dag: HTCondor accounting_group (default: %(default)s)")
    p.add_argument("--image", default=DEFAULT_IMAGE,
                   help="dag: the container image of the jobs, built from sage_cpu.def (default: %(default)s)")
    args = p.parse_args(argv)
    df = pd.read_csv(args.csv)
    if "GPStime" not in df.columns:
        p.error(f"{args.csv} has no GPStime column (a Git LFS pointer not pulled?): {list(df.columns)[:5]}")
    parts = n_parts(len(df), args.chunk)
    if args.command == "parts":
        print(parts)
        return
    if args.out is None:
        p.error(f"{args.command} needs OUT")
    if args.command == "dag":
        write_dag(args.csv, args.out, parts, args.chunk, args.accounting_group, args.tar or "virgo_glitches.tar",
                  args.image)
        return
    if args.command == "run":
        if not 0 <= args.part < parts:
            p.error(f"--part must be in 0 .. {parts - 1} for {len(df)} rows and --chunk {args.chunk}")
        name = part_name(args.part)
        if all((Path(args.out) / "parts" / f"{name}_{tag}.h5").exists() for tag in AVERAGES):
            print(f"part {args.part}: already done", flush=True)
            return
        lo = args.part * args.chunk
        print(f"part {args.part}/{parts}: rows {lo} .. {min(lo + args.chunk, len(df)) - 1}", flush=True)
        save_glitches(df.iloc[lo:lo + args.chunk], Path(args.out) / "parts", name=name)
    else:
        merge(args.out, parts, tar=args.tar)


if __name__ == "__main__":
    main()
