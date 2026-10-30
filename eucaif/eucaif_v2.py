"""
EuCAIF dataset v2: whitened H1/L1 strain mixing Gaussian noise, BBH signals and O3a-calibrated
analytic glitches, from one config (``eucaif_v2.yaml``), on the fly or written to HDF5.

It reuses v1's ``WhitenedExampleGenerator`` (same BBH prior, IMRPhenomPv2, projection, noise and
whitening; see README_dataset.md) and adds glitches from ``sage.data.noise.GlitchPopulation``,
coloured with the same ASD and added before whitening. The stored window is
``[q_padding | frame_length | q_padding]``; time series and Q-transforms are cut to the frame.

    from eucaif.eucaif_v2 import EucaifV2OnTheFly, QTransforms
    train = EucaifV2OnTheFly("eucaif/eucaif_v2.yaml", batch_size=64, batches_per_epoch=500, seed=0)
    qt = QTransforms("eucaif/eucaif_v2.yaml", device="cuda")
    for batch in DataLoader(train, batch_size=None, num_workers=8):
        x = batch["x"].cuda()                    # (B, 2, L) whitened H1/L1, stored window
        series = qt.frame(x)                     # (B, 2, frame samples): time-series input
        images = qt.qscan(x)                     # (B, 2, n_times, n_freqs): Q-scan input
        tiles = qt.tiles(x)                      # {Q: (B, 2, rows, cols) complex}: invertible input

Writing fixed sets (from a clean, tagged checkout, so the file records the version):

    PYTHONPATH=. python3 eucaif/eucaif_v2.py eucaif/eucaif_v2.yaml --n-per-class 5000 --seed 1 \\
        --out eucaif_v2_val.h5 --fig v2_check.png
"""

import sys
import tempfile
from pathlib import Path

SAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAGE_ROOT))
sys.path.insert(0, str(SAGE_ROOT / "eucaif"))

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset, get_worker_info

from make_whitened_examples import WhitenedExampleGenerator, write_dataset
from sage.core.config import register_configs
from sage.data.noise import GlitchPopulation
from sage.dsp.qtransform import QTransform

KEYS = ("x", "y", "glitch_whitened", "signal_whitened", "n_glitches", "snr", "snr_det", "tc")


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["_dir"] = path.resolve().parent
    cfg["_text"] = path.read_text()
    return cfg


def _register(cfg, batch_size=64, device="cpu"):
    """sage's global configs for the waveform generator (as make_whitened_examples' __main__)."""
    d = cfg["data"]
    stored = d["frame_length"] + 2 * d["q_padding"]

    class CFG:
        export_dir = "."
        dtype = torch.float32
        detectors = list(d["detectors"])
        class_balance = 0.5
        do_point_estimate = ["tc", "mchirp", "ra", "dec"]

    class DataCFG:
        sample_rate = float(d["sample_rate"])
        noise_low_frequency_cutoff = float(d["whitening_highpass"])
        signal_low_frequency_cutoff = float(d["signal_low_frequency_cutoff"])
        sample_length_in_s = float(stored)
        padding_length_in_s = float(d["signal_padding"])
        padded_length_in_s = float(stored + 2 * d["signal_padding"])
        delta_f = 1.0 / float(stored)
        corrupted_length = float(d["whitening_fduration"]) / 2

    CFG.batch_size = int(batch_size)
    CFG.device = device
    register_configs(CFG(), DataCFG())


def _prior_with_tc(cfg, tmpdir):
    """v1's prior with tc (from the start of the stored window) = q_padding + tc_in_frame."""
    prior = yaml.safe_load((cfg["_dir"] / cfg["signal"]["prior"]).read_text())
    lo, hi = cfg["signal"]["tc_in_frame"]
    q = cfg["data"]["q_padding"]
    prior["priors"]["tc"] = {"name": "uniform", "min": q + lo, "max": q + hi}
    path = Path(tmpdir) / "eucaif_v2_waveform.yaml"
    path.write_text(yaml.safe_dump(prior, sort_keys=False))
    return path


def _populations(cfg):
    """The O3a glitches, and the non-physical (McGAN) group if configured: two independent populations
    drawn from the same population file, with disjoint classes."""
    g = cfg["glitches"]
    base = yaml.safe_load((cfg["_dir"] / g["population"]).read_text())
    common = dict(detectors=list(cfg["data"]["detectors"]), sample_rate=float(cfg["data"]["sample_rate"]))
    extra = cfg.get("nonphysical_glitches") or {}
    names = list(extra.get("classes", []))
    unknown = set(names) - set(base["classes"])
    if unknown:
        raise ValueError(f"nonphysical_glitches: classes {sorted(unknown)} not in {g['population']}")
    physical = {k: v for k, v in base["classes"].items() if k not in names}
    pops = [GlitchPopulation(dict(base, classes=physical, glitches_per_segment=list(g["per_window"]),
                                  segment_probability=float(g["probability"]), **common))]
    if names and float(extra.get("probability", 1.0)) > 0:
        like = base["classes"][extra["snr_like"]] if extra.get("snr_like") else {}
        classes = {}
        for k in names:
            c = dict(base["classes"][k])
            c.pop("rate", None)
            c["weight"] = 1.0                                  # equally likely within the group
            if like:                                           # deliberately the SNR distribution of snr_like
                c.update(snr_range=like["snr_range"], snr_index=like["snr_index"])
            classes[k] = c
        pops.append(GlitchPopulation(dict(classes=classes, glitches_per_segment=list(extra.get("per_window", [1, 1])),
                                          segment_probability=float(extra.get("probability", 1.0)), **common)))
    return pops


def make_generator(config, seed=0, device="cpu", dtype=torch.float32, batch_size=64):
    """A ``WhitenedExampleGenerator`` set up from the v2 config (registers sage's global configs)."""
    cfg = load_config(config) if not isinstance(config, dict) else config
    _register(cfg, batch_size, device)
    d = cfg["data"]
    tmp = tempfile.TemporaryDirectory()            # holds the v2 prior while the generator lives
    gen = WhitenedExampleGenerator(
        _prior_with_tc(cfg, tmp.name), asd_names=d["asd"], snr_range=cfg["signal"].get("snr_range"),
        fduration=float(d["whitening_fduration"]), highpass=d["whitening_highpass"], dtype=dtype,
        seed=seed, merger_margin=d["q_padding"] + 0.25, glitches=_populations(cfg),
        glitch_f_low=cfg["glitches"].get("f_low"))
    gen._tmpdir = tmp
    gen.frame_slice = (int(round(d["q_padding"] * gen.fs)),
                       int(round((d["q_padding"] + d["frame_length"]) * gen.fs)))
    attrs = gen.attrs
    gen.attrs = lambda: {**attrs(), "eucaif_v2_config": cfg["_text"], "frame_slice": np.array(gen.frame_slice),
                         "dataset_version": "eucaif v2.2"}
    return gen


class EucaifV2OnTheFly(IterableDataset):
    """Fresh shuffled batches (half BBH, glitches as configured); use with ``DataLoader(batch_size=None)``.

    Batch ``i`` of epoch ``e`` is the same whatever the number of workers: its noise, glitches, BBH
    parameters and projection are all seeded from ``(seed, e, i)``.
    """

    def __init__(self, config, batch_size=64, batches_per_epoch=1000, seed=0, device="cpu", dtype=torch.float32):
        if batch_size % 2:
            raise ValueError("batch_size must be even (half BBH, half noise only)")
        self.config, self.batch_size, self.batches_per_epoch = str(config), int(batch_size), int(batches_per_epoch)
        self.seed, self.device, self.dtype, self.epoch = int(seed), device, dtype, 0
        self._gen = None
        cfg = load_config(config)
        d = cfg["data"]
        fs = float(d["sample_rate"])
        self.frame_slice = (int(round(d["q_padding"] * fs)), int(round((d["q_padding"] + d["frame_length"]) * fs)))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def batch_seed(self, epoch, index):
        return int(np.random.SeedSequence([self.seed, epoch, index]).generate_state(1, dtype=np.uint32)[0])

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        if self._gen is None:                      # in each worker: sage's configs are per process
            self._gen = make_generator(self.config, seed=self.seed, device=self.device, dtype=self.dtype)
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        for i in range(wid, self.batches_per_epoch, nw):
            s = self.batch_seed(self.epoch, i)
            self._gen.param_sampler.generator.manual_seed(s)
            data = self._gen.generate(self.batch_size // 2, noise_seed=s, keep_signal=True)
            order = torch.randperm(self.batch_size, generator=torch.Generator().manual_seed(s))
            n = self.batch_size // 2
            signal = torch.cat([torch.zeros_like(data["signal_whitened"]), data["signal_whitened"]])
            nan = torch.full((n,), float("nan"))
            yield {"x": data["x"][order].float(), "y": data["y"][order],
                   "glitch": data["glitch_whitened"][order].float(), "signal": signal[order].float(),
                   "n_glitches": torch.as_tensor(data["n_glitches"].astype(np.int64))[order],
                   "snr": torch.cat([nan, data["snr"].float()])[order],
                   "tc": torch.cat([nan, data["tc"].float()])[order]}


class QTransforms:
    """Time-series and Q-transform inputs from batches of stored windows (B, D, L), cut to the frame."""

    def __init__(self, config, device="cpu", dtype=torch.float32):
        cfg = load_config(config)
        d, q = cfg["data"], cfg["qtransform"]
        self.fs = float(d["sample_rate"])
        self.qpad, self.frame_s = float(d["q_padding"]), float(d["frame_length"])
        self.L = int(round((self.frame_s + 2 * self.qpad) * self.fs))
        self.frame_slice = slice(int(round(self.qpad * self.fs)), int(round((self.qpad + self.frame_s) * self.fs)))
        self.scan = QTransform(self.fs, self.L, qrange=tuple(q["qrange"]), frange=tuple(q["frange"]), logf=True,
                               fres=int(q["n_freqs"]), tres=self.frame_s / int(q["n_times"]),
                               outseg=(self.qpad, self.qpad + self.frame_s), dtype=dtype).to(device)
        self.tile_qt = {float(qq): QTransform(self.fs, self.L, qrange=(qq, qq), frange=tuple(q["frange"]),
                                              norm=False, dtype=dtype).to(device) for qq in q["tiles_q"]}

    def frame(self, x):
        """The frame of the time series: (..., L) -> (..., frame samples)."""
        return x[..., self.frame_slice]

    def qscan(self, x, return_q=False):
        """gwpy-exact Q-scan (median-normalised energy) of the stored window, on the frame:
        (..., L) -> (..., n_times, n_freqs)."""
        return self.scan(x, return_q=return_q)

    def tiles(self, x):
        """Invertible complex tiles of the stored window, one entry per Q: {Q: (..., rows, cols)}.
        ``inverse(tiles, Q)`` gives the time series back; cut its frame with ``frame``."""
        return {q: qt.tiles(x, q) for q, qt in self.tile_qt.items()}

    def inverse(self, tiles, q):
        return self.tile_qt[float(q)].inverse(tiles, q)


def plot_check(data, gen, config, path):
    """Four examples (noise; noise + glitches; BBH; BBH + glitches): Q-scans and time series, H1 and L1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qt = QTransforms(config)
    y = data["y"].numpy()
    ng = data["n_glitches"].sum(1)
    picks = []
    for want_y, want_g in ((0, False), (0, True), (1, False), (1, True)):
        idx = np.nonzero((y == want_y) & ((ng > 0) == want_g))[0]
        if len(idx):
            picks.append((int(idx[0]), want_y, want_g))
    D = gen.D
    fig, axes = plt.subplots(2 * D, len(picks), figsize=(4.2 * len(picks), 3.2 * D * 1.6), squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.4, 0.8] * D, hspace=0.55))
    t = gen.t.numpy() - qt.qpad
    fr = qt.frame_slice
    for j, (i, yy, gg) in enumerate(picks):
        x = data["x"][i].float()
        scan, q = qt.qscan(x, return_q=True)
        for d in range(D):
            ax = axes[2 * d, j]
            ax.pcolormesh(qt.scan.times.numpy() - qt.qpad, qt.scan.frequencies.numpy(), scan[d].T.numpy(),
                          shading="auto", vmin=0, vmax=25, rasterized=True)
            ax.set(yscale="log", title=f"{gen.detectors[d]}: {'BBH' if yy else 'noise'}"
                   f"{' + %d glitch(es)' % data['n_glitches'][i, d] if data['n_glitches'][i, d] else ''}"
                   f", Q = {float(q[d]):.0f}")
            ax.title.set_fontsize(9)
            if j == 0:
                ax.set_ylabel("frequency [Hz]")
            ax = axes[2 * d + 1, j]
            ax.plot(t[fr], x[d].numpy()[fr], lw=0.4, color="0.6", label="whitened strain")
            if gg:
                ax.plot(t[fr], data["glitch_whitened"][i, d].numpy()[fr], lw=0.8, color="#e34948", label="glitches")
            if yy:
                k = i - len(y) // 2
                ax.plot(t[fr], data["signal_whitened"][k, d].numpy()[fr], lw=0.9, color="#2a78d6", label="BBH")
                ax.axvline(float(data["tc"][k]) - qt.qpad, color="k", ls="--", lw=0.7)
            ax.legend(fontsize=6, loc="upper left")
            ax.set_xlabel("time from the frame start [s]", fontsize=8)
    fig.suptitle("EuCAIF v2 examples: Q-scans (frame) and whitened time series", y=0.995)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("saved", path)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--n-per-class", type=int, default=0, help="examples per class to write (0: check only)")
    p.add_argument("--chunk-per-class", type=int, default=500)
    p.add_argument("--out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--store-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--fig", help="check figure")
    args = p.parse_args()

    gen = make_generator(args.config, seed=args.seed,
                         dtype=torch.float64 if args.store_dtype == "float64" else torch.float32)
    data = gen.generate(32, noise_seed=args.seed + 999_999, keep_signal=True)
    noise = data["x"] - data["glitch_whitened"]
    noise[32:] -= data["signal_whitened"]
    print(f"stored window {gen.sample_length_s:g} s ({gen.L} samples x {gen.D} detectors at {gen.fs:g} Hz), "
          f"frame samples {gen.frame_slice}")
    print(f"whitened noise std {float(noise.std()):.3f} (expect ~1); BBH network SNR median "
          f"{float(data['snr'].median()):.1f}; examples x detectors with glitches "
          f"{100 * float((data['n_glitches'] > 0).mean()):.0f}%")
    tab = data["glitch_table"]
    classes = gen.glitch_classes
    print("glitch classes:", dict(zip(*[[classes[c] for c in np.unique(tab["class"])],
                                         np.unique(tab["class"], return_counts=True)[1].tolist()])))
    if args.fig:
        plot_check(data, gen, args.config, args.fig)
    if args.out and args.n_per_class:
        from make_whitened_examples import provenance
        if provenance()["sage_uncommitted_changes"]:
            print("WARNING: uncommitted changes in the sage checkout; the stored commit won't reproduce the file.")
        write_dataset(gen, args.out, args.n_per_class, chunk_per_class=args.chunk_per_class,
                      store_dtype=args.store_dtype)
        print("done:", args.out)


if __name__ == "__main__":
    main()
