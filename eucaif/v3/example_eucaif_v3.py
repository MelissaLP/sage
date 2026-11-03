"""EuCAIF v3 example on real data from the local cache: Q-scans and whitened time series of a noise-only and a BBH
example (first detector of the network), with the injected BBH and glitches.

    PYTHONPATH=. python eucaif/v3/example_eucaif_v3.py --out eucaif/v3/figures/eucaif_v3_example.png
    PYTHONPATH=. python eucaif/v3/example_eucaif_v3.py --cache /path/to/v3_cache --detectors H1   # a partial cache
"""
import argparse
import copy

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from eucaif.v3 import segments as S  # noqa: E402
from eucaif.v3.generator import EucaifV3, load_config  # noqa: E402
from sage.dsp.qtransform import QTransform  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--config", default="eucaif/v3/eucaif_v3.yaml")
p.add_argument("--cache", help="cache directory (default: the config's)")
p.add_argument("--detectors", nargs="+", help="a sub-network of the config's (its good time is a superset)")
p.add_argument("--snr", type=float, nargs=2, default=[15.0, 20.0], help="network SNR range of the BBHs")
p.add_argument("--seed", type=int, default=4)
p.add_argument("--out", default="eucaif_v3_example.png")
args = p.parse_args()

cfg = copy.deepcopy(load_config(args.config))
if args.cache:
    cfg["cache"] = dict(cfg["cache"], directory=args.cache)
if args.detectors:
    seg = S.load(cfg["_dir"] / cfg["segments"]["file"])
    seg_file = args.out.rsplit(".", 1)[0] + "_segments.json"
    S.save(dict(seg, detectors=args.detectors), seg_file)
    cfg["data"]["detectors"], cfg["segments"]["file"] = list(args.detectors), seg_file
cfg["cache"]["fetch_missing"] = False
cfg["split"] = dict(mode="random")
cfg["signal"]["snr_range"] = list(args.snr)
gen = EucaifV3(cfg, split="train", seed=args.seed, batch_size=8)
b = gen.batch(8)
fs, L, det = gen.fs, gen.L, gen.detectors[0]
ys = b["y"].numpy()
fewest = lambda idx: min(idx, key=lambda k: len(b["glitch_events"][k]))           # noqa: E731, clearest example
pos, neg = fewest(np.nonzero(ys == 1)[0]), fewest(np.nonzero(ys == 0)[0])
qt = QTransform(fs, L, frange=(15, 500), logf=True, outseg=(1.0, 3.0))           # the 2 s frame
t = np.arange(L) / fs - 1.0                                                      # from the frame start

fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), subplot_kw=dict(axes_class=Axes), sharex=True)
for col, i in enumerate((neg, pos)):
    img, q = qt(b["x"][i, 0], return_q=True)
    ax = axes[0, col]
    ax.pcolormesh(qt.times - 1.0, qt.frequencies, img.T, shading="auto", vmin=0, vmax=25, rasterized=True)
    label = "no BBH" if b["y"][i] == 0 else f"BBH, SNR {float(b['snr'][i]):.1f}, tc {float(b['tc'][i]) - 1:.2f} s"
    ax.set(yscale="log", ylabel="frequency [Hz]",
           title=f"{det} GPS {float(b['gps'][i]) + 1:.2f} (frame start), {label} - Q = {float(q):.1f}")
    ax = axes[1, col]
    sel = (t >= 0) & (t <= 2)
    ax.plot(t[sel], b["x"][i, 0].numpy()[sel], color="#b9b8b2", lw=0.5, label="whitened data")
    if b["y"][i] == 1:
        ax.plot(t[sel], b["signal"][i, 0].numpy()[sel], color="#2a78d6", lw=1, label="injected BBH")
    ax.plot(t[sel], b["glitch"][i, 0].numpy()[sel], color="#eb6834", lw=1, label="injected glitches")
    for e in b["glitch_events"][i]:
        tt = e["time"] - gen.stored_offset - 1.0
        if e["detector"] == det and 0 <= tt <= 2:
            ax.annotate(f"{e['cls']}\nSNR {e['snr']:.0f}", (tt, ax.get_ylim()[1] * 0.85), fontsize=7,
                        ha="center", color="#eb6834")
    ax.set(xlabel="time from the frame start [s]", ylabel="whitened strain", xlim=(0, 2))
    ax.legend(loc="lower left", fontsize=7)
src = b["sources"][neg][0]
fig.suptitle(f"EuCAIF v3: real {det} strain from the cache ({src}, decimated to {fs:.0f} Hz)\nSAGE "
             f"{cfg['signal'].get('approximant', 'IMRPhenomPv2')} BBH and analytic glitches at their SNR against "
             f"the window's own PSD; {cfg['whitening']['method']} whitening", fontsize=10)
fig.tight_layout()
fig.savefig(args.out, dpi=110)
print("saved", args.out, "| examples (y, SNR, glitches):",
      [(int(b["y"][k]), round(float(b["snr"][k]), 1) if b["y"][k] else None,
        [e["cls"] for e in b["glitch_events"][k]]) for k in range(8)])
