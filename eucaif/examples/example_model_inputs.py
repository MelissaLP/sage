"""The model inputs of a EuCAIF dataset version, as a network sees them: the whitened time series of the 2 s frame
and its Q-scan (``QTransforms.inputs(x, ["timeseries", "qscan"])``), for a noise-only and a BBH example, per
detector, with the injected BBH and glitches drawn on the time series.

    PYTHONPATH=. python eucaif/examples/example_model_inputs.py v1 --out eucaif/v1/figures/eucaif_v1_model_inputs.png
    PYTHONPATH=. python eucaif/examples/example_model_inputs.py v2 --out eucaif/v2/figures/eucaif_v2_model_inputs.png
    PYTHONPATH=. python eucaif/examples/example_model_inputs.py v3 --cache /path/to/v3_cache --detectors H1 \\
        --out eucaif/v3/figures/eucaif_v3_model_inputs.png

v1 stores only the 2 s frame (no ``q_padding``), so its Q-scan is computed on the frame itself and its edges
(about 0.1 s) carry the transform's circular wrap-around; v2 and v3 store 1 s more on each side for this.
"""
import argparse
import copy

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from eucaif.v2.eucaif_v2 import QTransforms  # noqa: E402

DATA, BBH, GLITCH = "#b9b8b2", "#2a78d6", "#eb6834"


def v1_examples(seed):
    """v1 (README_dataset.md): Gaussian noise + physical BBHs, 2 s windows, no glitches."""
    from eucaif.v2.eucaif_v2 import load_config
    from eucaif.v1.make_whitened_examples import WhitenedExampleGenerator
    from sage.core.config import register_configs

    class CFG:                                     # the configs of make_whitened_examples.py's __main__
        export_dir, batch_size, device, dtype = ".", 10, "cpu", torch.float32
        detectors, class_balance = ["H1", "L1"], 0.5
        do_point_estimate = ["tc", "mchirp", "ra", "dec"]

    class DataCFG:
        sample_rate, noise_low_frequency_cutoff, signal_low_frequency_cutoff = 2048.0, 15.0, 20.0
        sample_length_in_s, padded_length_in_s, padding_length_in_s = 2.0, 10.0, 4.0
        delta_f, corrupted_length = 0.5, 2.0

    register_configs(CFG(), DataCFG())
    gen = WhitenedExampleGenerator("eucaif/v1/sage_eucaif_waveform.yaml", seed=seed)
    n = 8
    d = gen.generate(n, noise_seed=seed, keep_signal=True)
    signal = torch.cat([torch.zeros_like(d["signal_whitened"]), d["signal_whitened"]])
    nan = torch.full((n,), float("nan"))
    qcfg = dict(data=dict(sample_rate=2048.0, q_padding=0.0, frame_length=2.0),
                qtransform=load_config("eucaif/v2/eucaif_v2.yaml")["qtransform"])
    return dict(x=d["x"].float(), y=d["y"], signal=signal.float(), glitch=torch.zeros_like(signal).float(),
                snr=torch.cat([nan, d["snr"].float()]), tc=torch.cat([nan, d["tc"].float()]),
                detectors=["H1", "L1"], qt=QTransforms(qcfg),
                title="EuCAIF v1: Gaussian noise (aLIGOZeroDetHighPower) + IMRPhenomPv2 BBH at physical distance")


def v2_examples(seed):
    """v2 (README_eucaif_v2.md): v1 + O3a analytic and McGAN glitches, 4 s stored windows."""
    from eucaif.v2.eucaif_v2 import make_generator
    config = "eucaif/v2/eucaif_v2.yaml"
    gen = make_generator(config, seed=seed)
    n = 8
    d = gen.generate(n, noise_seed=seed, keep_signal=True)
    signal = torch.cat([torch.zeros_like(d["signal_whitened"]), d["signal_whitened"]])
    nan = torch.full((n,), float("nan"))
    return dict(x=d["x"].float(), y=d["y"], signal=signal.float(), glitch=d["glitch_whitened"].float(),
                snr=torch.cat([nan, d["snr"].float()]), tc=torch.cat([nan, d["tc"].float()]),
                detectors=list(gen.detectors[:gen.D]), qt=QTransforms(config),
                title="EuCAIF v2: Gaussian noise + IMRPhenomPv2 BBH + O3a analytic and McGAN glitches")


def v3_examples(seed, cache=None, detectors=None):
    """v3 (v3/README.md): real strain from the local cache + BBH + glitches, 4 s stored windows."""
    from eucaif.v3 import segments as S
    from eucaif.v3.generator import EucaifV3, load_config
    cfg = copy.deepcopy(load_config("eucaif/v3/eucaif_v3.yaml"))
    if cache:
        cfg["cache"] = dict(cfg["cache"], directory=cache)
    if detectors:
        seg = S.load(cfg["_dir"] / cfg["segments"]["file"])
        import tempfile
        seg_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        S.save(dict(seg, detectors=detectors), seg_file)
        cfg["data"]["detectors"], cfg["segments"]["file"] = list(detectors), seg_file
    cfg["cache"]["fetch_missing"] = False
    cfg["split"] = dict(mode="random")
    cfg["signal"]["snr_range"] = [15.0, 20.0]
    gen = EucaifV3(cfg, split="train", seed=seed, batch_size=8, log=lambda *a: None)
    b = gen.batch(8)
    run = b["sources"][0][0]
    return dict(x=b["x"], y=b["y"], signal=b["signal"], glitch=b["glitch"], snr=b["snr"].float(),
                tc=b["tc"].float(), detectors=gen.detectors, qt=QTransforms(cfg), gps=b["gps"],
                title=f"EuCAIF v3: real strain ({run}, cached) + IMRPhenomPv2 BBH (SNR 15-20) + analytic and McGAN "
                      "glitches")


def pick(ex):
    """The noise-only example with the fewest glitches, and the loudest BBH."""
    y, g = ex["y"].numpy(), ex["glitch"].abs().amax(dim=(1, 2)).numpy()
    neg = np.nonzero(y == 0)[0]
    pos = np.nonzero(y == 1)[0]
    return int(neg[np.argmin(g[neg])]), int(pos[np.nanargmax(ex["snr"].numpy()[pos])])


def plot(ex, out):
    qt, dets = ex["qt"], ex["detectors"]
    inputs = qt.inputs(ex["x"], ["timeseries", "qscan"])          # what a model gets
    _, qs = qt.qscan(ex["x"], return_q=True)
    cut = lambda a: qt.frame(a)                                    # noqa: E731, the frame of a target
    D, fs = len(dets), qt.fs
    t = np.arange(inputs["timeseries"].shape[-1]) / fs
    times, freqs = qt.scan.times.numpy() - qt.qpad, qt.scan.frequencies.numpy()
    cols = pick(ex)
    height = 3.4 * D
    fig, axes = plt.subplots(2 * D, 2, figsize=(12, height), sharex=True, squeeze=False,
                             subplot_kw=dict(axes_class=Axes), gridspec_kw=dict(height_ratios=[0.8, 1.2] * D))
    for c, i in enumerate(cols):
        if ex["y"][i] == 1:
            what = f"BBH, network SNR {float(ex['snr'][i]):.1f}, merger at {float(ex['tc'][i]) - qt.qpad:.2f} s"
        else:
            what = "noise only (no BBH)"
        for d, det in enumerate(dets):
            ax = axes[2 * d, c]
            ax.plot(t, inputs["timeseries"][i, d].numpy(), color=DATA, lw=0.5)
            if ex["y"][i] == 1:
                ax.plot(t, cut(ex["signal"][i, d]).numpy(), color=BBH, lw=1.0)
            if ex["glitch"][i, d].abs().max() > 0:
                ax.plot(t, cut(ex["glitch"][i, d]).numpy(), color=GLITCH, lw=1.0)
            ax.set_title(f"{det}: {what}", fontsize=9)
            ax.set_ylabel("whitened strain", fontsize=8)
            ax = axes[2 * d + 1, c]
            im = ax.pcolormesh(times, freqs, inputs["qscan"][i, d].numpy().T, shading="auto", vmin=0, vmax=25,
                               rasterized=True)
            ax.set(yscale="log", ylabel="frequency [Hz]")
            ax.set_yticks([20, 50, 100, 200, 500], labels=["20", "50", "100", "200", "500"])
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.yaxis.label.set_fontsize(8)
            ax.text(0.01, 0.95, f'input "qscan" (Q = {float(qs[i, d]):.0f})', transform=ax.transAxes, va="top",
                    fontsize=7, color="white")
    for ax in axes[-1]:
        ax.set_xlabel("time from the frame start [s]")
    axes[0, 0].set_xlim(0, t[-1])
    fig.tight_layout()
    for d in range(D):                                             # outside the grid: keeps the time axes aligned
        box = axes[2 * d + 1, 1].get_position()
        cax = fig.add_axes([box.x1 + 0.01, box.y0, 0.01, box.height])
        fig.colorbar(im, cax=cax, label="normalised energy")
    handles = [plt.Line2D([], [], color=DATA, lw=1.5, label='input "timeseries" (whitened data)'),
               plt.Line2D([], [], color=BBH, lw=1.5, label="injected BBH, whitened alone"),
               plt.Line2D([], [], color=GLITCH, lw=1.5, label="injected glitches, whitened alone")]
    if not ex["glitch"].abs().max() > 0:
        handles = handles[:2]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0 + 0.05 / height))
    fig.suptitle(f"{ex['title']}\nModel inputs: QTransforms.inputs(x, [\"timeseries\", \"qscan\"]) on the 2 s frame",
                 fontsize=10, y=1.0 + 0.55 / height, va="bottom")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version", choices=["v1", "v2", "v3"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cache", help="v3: cache directory (default: the config's)")
    p.add_argument("--detectors", nargs="+", help="v3: a sub-network of the config's (e.g. a partial cache)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.version == "v3":
        ex = v3_examples(args.seed, args.cache, args.detectors)
    else:
        ex = dict(v1=v1_examples, v2=v2_examples)[args.version](args.seed)
    plot(ex, args.out or f"eucaif_{args.version}_model_inputs.png")
