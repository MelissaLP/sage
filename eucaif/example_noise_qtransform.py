"""
Minimal example: Gaussian noise with a LIGO ASD, whitened, and its Q-transform with SAGE.

    PYTHONPATH=. python3 eucaif/example_noise_qtransform.py
"""

import matplotlib.pyplot as plt
import torch

from sage.data.noise import sample_synthetic_noise
from sage.dsp.qtransform import QTransform
from sage.dsp.whiten import FIRWhitening

fs, duration = 2048.0, 8.0                       # sample rate (Hz) and length (s)

# 1. coloured Gaussian noise (one detector) and the ASD that coloured it
noise, asd = sample_synthetic_noise(duration, ["aLIGOZeroDetHighPower"], detectors=1, batch=1,
                                    sample_rate=fs, seed=0, is_asd=True)
noise = torch.as_tensor(noise).double().reshape(1, 1, -1)       # (batch, detector, time)

# 2. whiten (gwpy-equivalent); 1 s is dropped at each end (fduration / 2)
whiten = FIRWhitening(sample_rate=fs, fduration=2.0, highpass=15.0, dtype=torch.float64)
white = whiten(noise, torch.as_tensor(asd).double())[0, 0]      # 6 s, unit variance

# 3. Q-transform (gwpy-exact): normalised energy on a (time, frequency) grid
qt = QTransform(fs, white.shape[-1], qrange=(4, 64), frange=(15, 1000), logf=True, fres=200)
qscan, q = qt(white, return_q=True)                              # (time, frequency)

print(f"whitened noise std = {float(white.std()):.3f}, Q-scan {tuple(qscan.shape)}, Q = {float(q):.1f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(torch.arange(white.shape[-1]) / fs, white, lw=0.3)
ax1.set_ylabel("whitened strain")
ax2.pcolormesh(qt.times, qt.frequencies, qscan.T, shading="auto", vmin=0, vmax=25)
ax2.set(yscale="log", xlabel="time [s]", ylabel="frequency [Hz]", title=f"SAGE Q-transform, Q = {float(q):.1f}")
ax2.set_xlim(0, white.shape[-1] / fs)
fig.savefig("noise_qtransform.png", dpi=120, bbox_inches="tight")
print("saved noise_qtransform.png")
