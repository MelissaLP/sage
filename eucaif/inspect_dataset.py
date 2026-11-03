"""
Inspect and check EuCAIF HDF5 files (v1, v2 and glitch frames; the kind is read from the file).

    python eucaif/inspect_dataset.py summary  FILE           # provenance, size, classes, SNRs, glitches
    python eucaif/inspect_dataset.py check    FILE           # consistency checks; exit code 1 if one fails
    python eucaif/inspect_dataset.py example  FILE 12 [--fig ex12.png]   # one example (+ time series, Q-scan)

``check`` reads the data in chunks, so it works on files larger than memory; the whitened-noise test uses a
sample of up to ``--n-sample`` examples.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

CHUNK = 256


def _str(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def kind(f):
    """'v1', 'v2' (or later 'v3'), or 'glitch-frames'."""
    if "dataset_version" in f.attrs:
        return _str(f.attrs["dataset_version"]).split()[-1].split(".")[0]
    if "frame_length_in_s" in f.attrs and "glitches/frame" in f:
        return "glitch-frames"
    return "v1"


def _classes(f):
    return [_str(c) for c in f.attrs.get("glitch_classes", f.attrs.get("classes", []))]


def _frame_slice(f):
    if "frame_slice" in f.attrs:
        return tuple(int(v) for v in f.attrs["frame_slice"])
    return 0, f["x"].shape[-1]                                     # v1: the stored window is the frame


def _quantiles(a, q=(0.1, 0.5, 0.9)):
    a = a[np.isfinite(a)]
    return "n/a" if a.size == 0 else " / ".join(f"{v:.1f}" for v in np.quantile(a, q))


def summary(f, path):
    N, D, L = f["x"].shape
    fs = float(f.attrs["sample_rate"])
    lines = [f"{path}  ({Path(path).stat().st_size / 1e9:.2f} GB)",
             f"kind: {kind(f)}" + (f"  ({_str(f.attrs['dataset_version'])})" if "dataset_version" in f.attrs else ""),
             f"code: {_str(f.attrs.get('sage_version', '?'))}  commit {_str(f.attrs.get('sage_commit', '?'))[:10]}"
             f"{'  (UNCOMMITTED CHANGES)' if f.attrs.get('sage_uncommitted_changes') else ''}",
             f"made: {_str(f.attrs.get('created_utc', '?'))}  seed {f.attrs.get('seed', '?')}",
             f"command: {_str(f.attrs.get('generation_command', '?'))}",
             f"x: {N} examples x {D} detectors ({', '.join(_str(d) for d in f.attrs.get('detectors', []))}) x {L} "
             f"samples ({L / fs:g} s at {fs:g} Hz), {f['x'].dtype}; frame samples {_frame_slice(f)}"]
    if "metadata/class" in f:
        y = f["metadata/class"][:]
        snr = f["metadata/snr"][:]
        lines.append(f"classes: {int((y == 0).sum())} noise only, {int((y == 1).sum())} BBH")
        lines.append(f"BBH network SNR 10/50/90%: {_quantiles(snr[y == 1])}")
    if "glitches/snr" in f:
        cls = f["glitches/class"][:]
        names = _classes(f)
        counts = np.bincount(cls, minlength=len(names))
        lines.append(f"glitches: {len(cls)} in {int((f['n_glitches'][:] > 0).sum())} of {N * D} "
                     f"(example, detector) pairs")
        snr = f["glitches/snr"][:]
        inf = f["glitches/snr_in_frame"][:] if "glitches/snr_in_frame" in f else None
        for c, name in enumerate(names):
            m = cls == c
            row = f"  {name:22s} {counts[c]:7d}   SNR 10/50/90%: {_quantiles(snr[m]):>18s}"
            if inf is not None:
                row += f"   in frame: {_quantiles(inf[m]):>18s}"
            lines.append(row)
    print("\n".join(lines))


def check(f, n_sample=200, seed=0):
    """[(status, message)], status in PASS / WARN / FAIL."""
    out = []
    ok = lambda cond, msg, bad="FAIL": out.append(("PASS" if cond else bad, msg))        # noqa: E731
    N, D, L = f["x"].shape
    ok(f["t"].shape == (L,), f"t has the window's {L} samples")
    for name in ("glitch",):
        if name in f:
            ok(f[name].shape == (N, D, L), f"{name} has the shape of x")
    for k in [k for k in f.get("metadata", {}).keys()]:
        ok(f[f"metadata/{k}"].shape[0] == N, f"metadata/{k} has {N} rows")

    finite = True
    for a in range(0, N, CHUNK):
        finite &= bool(np.isfinite(f["x"][a:a + CHUNK]).all())
        if "glitch" in f:
            finite &= bool(np.isfinite(f["glitch"][a:a + CHUNK]).all())
    ok(finite, "x (and glitch) are finite everywhere")

    if "metadata/class" in f:
        y = f["metadata/class"][:]
        n_per = f.attrs.get("n_per_class")
        ok(n_per is None or ((y == 0).sum() == n_per and (y == 1).sum() == n_per),
           f"classes balanced ({int((y == 0).sum())} / {int((y == 1).sum())})")
        snr, tc = f["metadata/snr"][:], f["metadata/tc"][:]
        ok(bool(np.isnan(snr[y == 0]).all() and np.isfinite(snr[y == 1]).all()), "BBH SNR set exactly for class 1")
        win = L / float(f.attrs["sample_rate"])
        ok(bool(((tc[y == 1] > 0) & (tc[y == 1] < win)).all()), f"every merger time tc inside the {win:g} s window")

    # whitened noise ~ unit variance: x minus the injected components, on noise-only examples
    rng = np.random.default_rng(seed)
    idx = np.arange(N) if "metadata/class" not in f else np.nonzero(f["metadata/class"][:] == 0)[0]
    idx = np.sort(rng.choice(idx, size=min(n_sample, idx.size), replace=False))
    if idx.size:
        noise = f["x"][idx].astype(np.float64)
        if "glitch" in f:
            noise -= f["glitch"][idx]
        s0, s1 = _frame_slice(f)
        std = float(noise[..., s0:s1].std())
        ok(abs(std - 1) < 0.05, f"whitened noise std in the frame {std:.3f} (expect 1 +- 0.05, "
                                f"{idx.size} examples)", bad="WARN")

    if "glitches/snr" in f:
        cols = {k: f[f"glitches/{k}"][:] for k in f["glitches"].keys()}
        n = {len(v) for v in cols.values()}
        ok(len(n) == 1, f"glitch table columns all {n.pop() if len(n) == 1 else n} rows")
        ex = cols.get("example", cols.get("frame"))
        counts = np.zeros((N, D), dtype=int)
        np.add.at(counts, (ex, cols["detector"]), 1)
        ok(bool((counts == f["n_glitches"][:]).all()), "glitch table matches n_glitches")
        ok(bool((cols["class"] < len(_classes(f))).all()), "glitch classes known (attrs)")
        ok(bool((cols["snr"] > 0).all()), "glitch SNRs positive")
        if "snr_in_frame" in cols:
            ok(bool((cols["snr_in_frame"] <= cols["snr"] * (1 + 1e-9)).all()
                    and ((cols["frame_fraction"] >= 0) & (cols["frame_fraction"] <= 1 + 1e-9)).all()),
               "in-frame glitch SNR <= SNR, frame fraction in [0, 1]")
        else:
            out.append(("WARN", "no in-frame glitch SNR (file written before snr_in_frame was added)"))
    dirty = bool(f.attrs.get("sage_uncommitted_changes", False))
    ok(not dirty, "written from a clean checkout (reproducible from its commit)" if not dirty else
       "written with uncommitted changes: its commit does not reproduce it exactly", bad="WARN")
    return out


def example(f, i, fig=None):
    N, D, L = f["x"].shape
    if not 0 <= i < N:
        raise SystemExit(f"example {i} not in [0, {N})")
    fs = float(f.attrs["sample_rate"])
    dets = [_str(d) for d in f.attrs.get("detectors", [])] or [f"det{d}" for d in range(D)]
    print(f"example {i} of {N}")
    for k in f.get("metadata", {}).keys():
        print(f"  {k:10s} {np.round(np.asarray(f[f'metadata/{k}'][i], dtype=np.float64), 4)}")
    rows = []
    if "glitches/snr" in f:
        col = "example" if "glitches/example" in f else "frame"
        m = np.nonzero(f[f"glitches/{col}"][:] == i)[0]
        names = _classes(f)
        for r in m:
            g = {k: f[f"glitches/{k}"][r] for k in ("detector", "class", "time", "snr", "snr_in_frame")
                 if f"glitches/{k}" in f}
            rows.append(g)
            print(f"  glitch: {dets[g['detector']]} {names[g['class']]:20s} t {g['time']:.3f} s  SNR {g['snr']:.1f}"
                  + (f"  in frame {g['snr_in_frame']:.1f}" if "snr_in_frame" in g else ""))
    if not fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import yaml
    from eucaif.v2.eucaif_v2 import QTransforms
    s0, s1 = _frame_slice(f)
    qcfg = dict(data=dict(sample_rate=fs, q_padding=s0 / fs, frame_length=(s1 - s0) / fs),
                qtransform=yaml.safe_load((Path(__file__).parent / "v2" / "eucaif_v2.yaml").read_text())["qtransform"])
    if s0 + s1 != L:
        raise SystemExit("the frame is not centred in the stored window; no Q-scan")
    qt = QTransforms(qcfg)
    x = torch.tensor(f["x"][i], dtype=torch.float32)
    img, q = qt.qscan(x, return_q=True)
    t = np.arange(s1 - s0) / fs
    figure, axes = plt.subplots(2 * D, 1, figsize=(9, 3.2 * D), sharex=True, squeeze=False,
                                gridspec_kw=dict(height_ratios=[0.8, 1.2] * D))
    for d in range(D):
        ax = axes[2 * d, 0]
        ax.plot(t, x[d, s0:s1].numpy(), color="#b9b8b2", lw=0.5, label="x")
        if "glitch" in f:
            ax.plot(t, f["glitch"][i, d, s0:s1], color="#eb6834", lw=1.0, label="glitch")
        ax.set(ylabel="whitened", title=f"{Path(f.filename).name} #{i}: {dets[d]}")
        ax.legend(loc="lower left", fontsize=7)
        ax = axes[2 * d + 1, 0]
        ax.pcolormesh(qt.scan.times.numpy() - s0 / fs, qt.scan.frequencies.numpy(), img[d].numpy().T,
                      shading="auto", vmin=0, vmax=25, rasterized=True)
        ax.set(yscale="log", ylabel="frequency [Hz]")
        ax.text(0.01, 0.95, f"Q-scan, Q = {float(q[d]):.0f}", transform=ax.transAxes, va="top", fontsize=7,
                color="white")
    axes[-1, 0].set(xlabel="time from the frame start [s]", xlim=(0, t[-1]))
    figure.tight_layout()
    figure.savefig(fig, dpi=110)
    print("saved", fig)


def main(argv=None):
    import h5py
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["summary", "check", "example"])
    p.add_argument("file")
    p.add_argument("index", nargs="?", type=int, help="example: the example's index")
    p.add_argument("--fig", help="example: save its time series and Q-scan here")
    p.add_argument("--n-sample", type=int, default=200, help="check: examples sampled for the noise test")
    args = p.parse_args(argv)
    with h5py.File(args.file, "r") as f:
        if args.command == "summary":
            summary(f, args.file)
        elif args.command == "check":
            results = check(f, args.n_sample)
            for status, msg in results:
                print(f"{status:4s}  {msg}")
            failed = sum(s == "FAIL" for s, _ in results)
            print(f"{failed} failed, {sum(s == 'WARN' for s, _ in results)} warnings")
            return 1 if failed else 0
        else:
            if args.index is None:
                p.error("example needs an index")
            example(f, args.index, args.fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
