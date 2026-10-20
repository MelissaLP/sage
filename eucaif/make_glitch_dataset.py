"""
Generate whitened frames full of analytic glitches (DeepExtractor-style) and store them in HDF5,
with the whitened glitches alone as the target and every glitch's class, time, SNR and parameters.

Everything is set in one config (``sage_analytic_glitches_dense.yaml``): the glitch classes and
how many per frame (see ``sage.data.noise.GlitchPopulation``), and the ``dataset`` block.

Pipeline, per frame and detector
--------------------------------
1. Coloured Gaussian noise (``sample_synthetic_noise``, one analytic ASD per detector) over
   ``frame_length + 2 * q_padding + whitening_fduration`` seconds.
2. Glitches (``GlitchPopulation.inject``) coloured with the same ASD, amplitude set by their optimal
   SNR against it, placed only inside the stored window (never in the edges the whitening drops).
3. ``FIRWhitening`` (gwpy-equivalent) of strain = noise + glitches, and of the glitches alone. The
   whitening is linear, so whitened strain = whitened noise + whitened glitches.
4. Stored window: ``[q_padding | frame_length | q_padding]``. Compute Q-transforms (e.g.
   ``QTransform.tiles``) on all of it and use the central frame: the transform is circular, so the
   padding absorbs its wrap-around.

HDF5 layout
-----------
    x            (n, D, L)  whitened strain
    glitch       (n, D, L)  whitened glitches alone (the target); whitened noise = x - glitch
    t            (L,)       time in s from the start of the frame (negative in the left padding)
    n_glitches   (n, D)     glitches per frame and detector
    glitches/    one row per glitch: frame, detector (index), class (index), time (s from the frame
                 start, the glitch centre or a ringdown's start), snr, and every generator parameter
                 (NaN where a class does not have it)
    attrs        classes, detectors, config text, dataset settings, seed, command, code version

Usage, from the SAGE repo root:

    PYTHONPATH=. python3 eucaif/make_glitch_dataset.py eucaif/sage_analytic_glitches_dense.yaml \\
        --n-frames 100000 --out glitch_frames.h5 [--device cuda] [--fig check.png]

Without ``--out`` it only runs the checks on a small chunk (and plots it with ``--fig``).
"""

import argparse
import math
import sys
import time
from pathlib import Path

SAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAGE_ROOT))
sys.path.insert(0, str(SAGE_ROOT / "eucaif"))

import numpy as np
import torch
import yaml

from sage.data.noise import GlitchPopulation, sample_synthetic_noise
from sage.dsp.whiten import FIRWhitening


class GlitchDatasetGenerator:
    """Set up the glitch population and the whitener once, then make chunks of frames with ``generate``."""

    def __init__(self, config, seed=0, device="cpu", dtype=torch.float32):
        self.config_path = str(config)
        self.config_text = Path(config).read_text()
        cfg = yaml.safe_load(self.config_text)
        ds = cfg["dataset"]
        self.pop = GlitchPopulation(cfg)
        self.detectors = self.pop.detectors
        self.D = len(self.detectors)
        self.asd_names = list(ds["asd"])
        if len(self.asd_names) != self.D:
            raise ValueError(f"{len(self.asd_names)} ASDs for {self.D} detectors")
        self.fs = self.pop.sample_rate
        self.frame_s = float(ds["frame_length"])
        self.qpad_s = float(ds.get("q_padding", 0.5))
        self.fduration = float(ds.get("whitening_fduration", 2.0))
        self.highpass = ds.get("whitening_highpass")
        self.f_low = ds.get("glitch_f_low")
        self.seed, self.device, self.dtype = int(seed), torch.device(device), dtype

        self.stored_s = self.frame_s + 2 * self.qpad_s
        self.L = int(round(self.stored_s * self.fs))
        self.window_s = self.stored_s + self.fduration
        self.N = int(round(self.window_s * self.fs))
        asd = sample_synthetic_noise(self.window_s, self.asd_names, detectors=self.D, batch=1,
                                     sample_rate=self.fs, seed=0, is_asd=True)[1]
        self.asd = torch.as_tensor(asd, dtype=torch.float64)
        self.whitener = FIRWhitening(sample_rate=self.fs, fduration=self.fduration, highpass=self.highpass,
                                     asd=self.asd, seq_len=self.N, dtype=dtype).to(self.device)
        if self.N - 2 * self.whitener.pad != self.L:
            raise ValueError(f"whitened length {self.N - 2 * self.whitener.pad} != stored length {self.L}")
        # glitches only where the whitened output is kept
        self.glitch_window = (self.fduration / 2, self.fduration / 2 + self.stored_s)
        self.t = torch.arange(self.L, dtype=torch.float64) / self.fs - self.qpad_s

        self.classes = list(self.pop.classes)
        fields = set()
        for name in self.classes:
            fields |= set(self.pop.sample(name, 1, torch.Generator().manual_seed(0))[1])
        self.param_fields = sorted(fields - {"snr"})

    def generate(self, n, chunk_seed):
        """``n`` frames: whitened strain and glitches (n, D, L), glitches per frame (n, D), glitch table."""
        noise = sample_synthetic_noise(self.window_s, self.asd_names, detectors=self.D, batch=n,
                                       sample_rate=self.fs, seed=int(chunk_seed), is_asd=True,
                                       device=self.device)[0]
        noise = torch.as_tensor(noise, device=self.device).to(self.dtype).reshape(n, self.D, self.N)
        g = torch.Generator().manual_seed(int(chunk_seed))
        glitches, events = self.pop.inject(torch.zeros_like(noise), self.asd.to(self.device), g,
                                           f_low=self.f_low, window=self.glitch_window)
        x = self.whitener(noise + glitches)
        target = self.whitener(glitches)

        rows = {k: [] for k in ["frame", "detector", "class", "time", "snr"] + self.param_fields}
        counts = np.zeros((n, self.D), dtype=np.int16)
        t0 = self.fduration / 2 + self.qpad_s            # frame start, in s from the generated start
        for i, evs in enumerate(events):
            frame, det = divmod(i, self.D)
            counts[frame, det] = len(evs)
            for e in evs:
                rows["frame"].append(frame)
                rows["detector"].append(det)
                rows["class"].append(self.classes.index(e["cls"]))
                rows["time"].append(e["time"] - t0)
                rows["snr"].append(e["snr"])
                for k in self.param_fields:
                    rows[k].append(e["params"].get(k, np.nan))
        table = {k: np.asarray(v, dtype=np.int32 if k in ("frame", "detector", "class") else np.float64)
                 for k, v in rows.items()}
        return x.cpu(), target.cpu(), counts, table

    def attrs(self):
        from make_whitened_examples import provenance   # code version (imports the waveform stack)
        return {
            "detectors": np.array(self.detectors, dtype="S"),
            "classes": np.array(self.classes, dtype="S"),
            "asd_names": np.array(self.asd_names, dtype="S"),
            "sample_rate": self.fs,
            "frame_length_in_s": self.frame_s,
            "q_padding_in_s": self.qpad_s,
            "stored_length_in_s": self.stored_s,
            "generated_length_in_s": self.window_s,
            "frame_slice": np.array([int(round(self.qpad_s * self.fs)),
                                     int(round((self.qpad_s + self.frame_s) * self.fs))]),
            "whitening": "FIRWhitening (gwpy 3.0.14 TimeSeries.whiten equivalent)",
            "whitening_fduration": self.fduration,
            "whitening_highpass": float(self.highpass) if self.highpass is not None else -1.0,
            "glitch_f_low": float(self.f_low) if self.f_low is not None else -1.0,
            "snr_definition": "optimal SNR of each glitch against its detector's colouring ASD",
            "time_reference": "seconds from the start of the frame (t, glitches/time)",
            "config_yaml": self.config_text,
            "seed": self.seed,
            **provenance(),
        }


def write_dataset(gen, out_path, n_frames, chunk=500, store_dtype="float32"):
    import h5py

    D, L = gen.D, gen.L
    attrs = gen.attrs()          # before opening the file (it imports modules and runs git)
    with h5py.File(out_path, "w") as f:
        dx = f.create_dataset("x", shape=(n_frames, D, L), dtype=store_dtype, chunks=(min(32, n_frames), D, L))
        dg = f.create_dataset("glitch", shape=(n_frames, D, L), dtype=store_dtype, chunks=(min(32, n_frames), D, L))
        dn = f.create_dataset("n_glitches", shape=(n_frames, D), dtype="int16")
        f.create_dataset("t", data=gen.t.numpy())
        cols = ["frame", "detector", "class", "time", "snr"] + gen.param_fields
        dt = {k: ("int32" if k in ("frame", "detector", "class") else "float64") for k in cols}
        dcol = {k: f.create_dataset(f"glitches/{k}", shape=(0,), maxshape=(None,), dtype=dt[k], chunks=(65536,))
                for k in cols}
        f.attrs["generation_command"] = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
        f.attrs["chunk"] = chunk
        f.attrs["store_dtype"] = store_dtype
        for k, v in attrs.items():
            f.attrs[k] = v

        written, k, t_start = 0, 0, time.perf_counter()
        while written < n_frames:
            n = min(chunk, n_frames - written)
            x, target, counts, table = gen.generate(n, chunk_seed=gen.seed * 1_000_003 + k)
            sl = slice(written, written + n)
            dx[sl] = x.numpy().astype(store_dtype)
            dg[sl] = target.numpy().astype(store_dtype)
            dn[sl] = counts
            m = len(table["frame"])
            for c in cols:
                ds = dcol[c]
                old = ds.shape[0]
                ds.resize((old + m,))
                ds[old:] = table[c] + (written if c == "frame" else 0)
            written += n
            k += 1
            rate = written / (time.perf_counter() - t_start)
            print(f"  {written}/{n_frames} frames ({rate:.1f} frames/s)", flush=True)
    return out_path


def check(gen, n=16, fig=None):
    """Sanity checks on one chunk; optional figure of the first frame."""
    x, target, counts, table = gen.generate(n, chunk_seed=gen.seed * 1_000_003 + 999_999)
    noise_w = (x - target).double()
    print(f"stored window {gen.stored_s:g} s = {gen.qpad_s:g} s pad + {gen.frame_s:g} s frame + "
          f"{gen.qpad_s:g} s pad ({gen.L} samples x {gen.D} detectors); generated {gen.window_s:g} s")
    print(f"whitened noise (x - glitch) std = {noise_w.std():.3f} (expect ~1)")
    print(f"glitches per frame and detector: {counts.min()}-{counts.max()} (mean {counts.mean():.1f}); "
          f"SNR {table['snr'].min():.1f}-{table['snr'].max():.1f}")
    classes, number = np.unique(table["class"], return_counts=True)
    print("classes:", ", ".join(f"{gen.classes[c]} {m}" for c, m in zip(classes, number)))
    lo, hi = table["time"].min(), table["time"].max()
    print(f"glitch times {lo:.2f}..{hi:.2f} s from the frame start (stored window {gen.t[0]:.2f}..{gen.t[-1]:.2f} s)")
    if fig:
        plot_frame(gen, x[0], target[0], table, fig)


def plot_frame(gen, x, target, table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sage.dsp.qtransform import QTransform

    qt = QTransform(gen.fs, gen.L, qrange=(4, 64), frange=(10, 1000), logf=True, fres=300,
                    tres=gen.stored_s / 1000)
    qscan, q = qt(x, return_q=True)
    t = gen.t.numpy()
    fig, axes = plt.subplots(2 * gen.D, 1, figsize=(13, 4.2 * gen.D), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, 0.8] * gen.D, hspace=0.15))
    for d, det in enumerate(gen.detectors):
        ax = axes[2 * d]
        ax.pcolormesh(qt.times.numpy() - gen.qpad_s, qt.frequencies.numpy(), qscan[d].T.numpy(),
                      shading="auto", vmin=0, vmax=25, rasterized=True)
        ax.set(yscale="log", ylabel="frequency [Hz]")
        ax.set_title(f"{det}: whitened strain, SAGE Q-transform (Q = {float(q[d]):.1f}) of the stored window",
                     fontsize=10, loc="left")
        ax = axes[2 * d + 1]
        ax.plot(t, x[d].numpy(), lw=0.3, color="0.65", label="whitened strain")
        ax.plot(t, target[d].numpy(), lw=0.7, color="#e34948", label="whitened glitches (target)")
        sel = (table["frame"] == 0) & (table["detector"] == d)
        for tt in table["time"][sel]:
            ax.axvline(tt, color="#2a78d6", lw=0.5, alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("whitened")
        for a in axes[2 * d:2 * d + 2]:
            for edge in (0.0, gen.frame_s):
                a.axvline(edge, color="white" if a is axes[2 * d] else "k", ls="--", lw=1)
    axes[-1].set_xlabel("time from the frame start [s] (dashed: frame edges)")
    axes[-1].set_xlim(t[0], t[-1])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print("saved", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--n-frames", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="whitening precision")
    p.add_argument("--store-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--fig", help="plot the first frame of the check chunk")
    args = p.parse_args()

    gen = GlitchDatasetGenerator(args.config, seed=args.seed, device=args.device,
                                 dtype=torch.float64 if args.dtype == "float64" else torch.float32)
    check(gen, fig=args.fig)
    from make_whitened_examples import provenance
    prov = provenance()
    if prov["sage_uncommitted_changes"]:
        print("WARNING: the sage checkout has uncommitted changes; the commit stored in the file will "
              "not reproduce this dataset exactly. Commit first.")
    if args.out:
        size = 2 * args.n_frames * gen.D * gen.L * np.dtype(args.store_dtype).itemsize
        print(f"writing {args.n_frames} frames to {args.out} (~{size / 1e9:.1f} GB + glitch table)")
        write_dataset(gen, args.out, args.n_frames, args.chunk, args.store_dtype)
        print("done:", args.out)


if __name__ == "__main__":
    main()
