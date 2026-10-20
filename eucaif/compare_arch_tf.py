"""
Scattered-light arch in 8 s of white noise: DeepExtractor STFT vs QTAM vs SAGE, same colour scale.

Brightness = pixel energy / mean noise energy of its frequency row (20 noise-only draws).
QTAM and SAGE use the same single Q; SAGE without median normalisation (raw energy).
The STFT uses DeepExtractor's settings (n_fft 512, Hann window 64, hop 32).

Usage, from the SAGE repo root (the QTAM panel needs ``pip install --no-deps qtam``):

    PYTHONPATH=. python3 eucaif/compare_arch_tf.py                       # -> arch_stft_qtam_sage.png
    PYTHONPATH=. python3 eucaif/compare_arch_tf.py --out arch.png
"""
import argparse
import math
import warnings

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
try:
    # qtam 0.3.1 imports its bundled torch_spline_interpolation_1_0_0 as a top-level module
    # (see benchmark_qtransform.py), so put the package's own folder on the path first
    import importlib.util
    import os
    import sys
    sys.path.insert(0, os.path.dirname(importlib.util.find_spec("qtam").origin))
    from qtam import SingleQTransform
except Exception:
    SingleQTransform = None

from sage.data.noise import GlitchInjector, scattered_light_arch
from sage.dsp.qtransform import QTransform

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", default="arch_stft_qtam_sage.png")
args = p.parse_args()

FS, DUR, SNR, Q = 4096.0, 8.0, 20.0, 45.3
F_ARCH, T_ARCH = 40.0, 4.0
FMIN, FMAX = 10.0, 200.0
N = int(FS * DUR)
torch.manual_seed(0)

asd = torch.full((N // 2 + 1,), math.sqrt(2.0 / FS), dtype=torch.float64)
arch = scattered_light_arch(1, FS, DUR, fringe_frequency=F_ARCH, timeperiod=T_ARCH, dtype=torch.float64)
h = GlitchInjector(FS, f_low=FMIN)(torch.zeros(1, 1, N, dtype=torch.float64), arch.view(1, 1, N), asd,
                                   snr=SNR, return_glitch=True)[1][:, 0]
noise = torch.randn(21, N, dtype=torch.float64)          # [0] for the image, [1:] for the noise floor
x = noise[:1] + h

# DeepExtractor: n_fft 512, Hann win 64, hop 32 (energy = magnitude^2)
win = torch.hann_window(64, dtype=torch.float64)
stft = lambda y: torch.stft(y, n_fft=512, hop_length=32, win_length=64, window=win, return_complex=True).abs() ** 2
# QTAM, single Q, full resolution
qtam = SingleQTransform(DUR, FS, q=Q, frange=[FMIN, float("inf")]) if SingleQTransform else None
qt = lambda y: torch.cat([qtam(c[:, None], phase_mode=False)[:, 0, 0] for c in y.split(5)])
# SAGE (gwpy-exact), the same single Q, raw energy on a regular grid
sage = QTransform(FS, N, qrange=(Q, Q), frange=(FMIN, 1000.0), norm=False, logf=True, fres=400, tres=0.01)
sg = lambda y: sage(y).transpose(-1, -2)                 # (B, F, T)

methods = [("DeepExtractor STFT\n(n_fft 512, win 64, hop 32)", stft, None, np.fft.rfftfreq(512, 1 / FS))]
if qtam is not None:
    methods.append((f"QTAM, Q = {Q}", qt, None,
                    np.asarray(qtam.freqs.cpu() if torch.is_tensor(qtam.freqs) else qtam.freqs)))
else:
    print("qtam not installed: QTAM panel skipped")
methods.append((f"SAGE (gwpy-exact), Q = {Q}", sg, sage.times.numpy(), sage.frequencies.numpy()))

panels = []
for name, fn, t, f in methods:
    floor = fn(noise[1:]).mean((0, -1))[:, None]
    img = (fn(x)[0] / floor).numpy()
    if t is None:
        t = np.arange(img.shape[-1]) * DUR / img.shape[-1]
    band = (f >= FMIN) & (f <= FMAX)
    noise_peak = float((fn(noise[1:2])[0] / floor)[torch.as_tensor(band)].max())
    panels.append((name, t, f, img, float(img[band].max()), noise_peak))
    print(f"{name.splitlines()[0]:28s} arch+noise peak {panels[-1][4]:6.1f} | noise-only peak {noise_peak:5.1f}")

fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 4.5), sharey=True)
for ax, (name, t, f, img, peak, npk) in zip(axes, panels):
    m = ax.pcolormesh(t, f, img, shading="auto", vmin=0, vmax=25, rasterized=True)
    ax.set(ylim=(FMIN, FMAX), yscale="log", xlabel="time [s]",
           title=f"{name}\npeak {peak:.0f} (noise-only peak {npk:.0f})")
axes[0].set_ylabel("frequency [Hz]")
fig.colorbar(m, ax=axes, label="energy / mean noise energy of the row", fraction=0.02)
fig.suptitle(f"Scattered-light arch ({F_ARCH:g} Hz, {T_ARCH:g} s, SNR {SNR:g}) in {DUR:g} s of white noise "
             f"at {FS:g} Hz, same noise and colour scale", y=1.12)
fig.savefig(args.out, dpi=110, bbox_inches="tight")
print("saved", args.out)
