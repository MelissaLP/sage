"""
Where does the scattered-light residual come from, and what reduces it?

40 Hz, 4 s arch at SNR 20 in 8 s of white noise (5 seeds), extracted from SAGE's invertible
Q-tiles with oracle masks (built from the glitch alone):

- binary mask at Q = 45, and its error split into noise kept inside the mask and glitch lost
  outside it;
- soft (Wiener) mask, best single Q, and a multi-Q pursuit (plane by plane, high Q first);
- a template fit with the true arch shape (least-squares amplitude and phase: no mask at all);
- how fast the template fit degrades when its parameters are slightly wrong.

Residual SNR = optimal SNR of (estimate - injected glitch): 0 is perfect, 20 is no subtraction.

Usage, from the SAGE repo root:

    PYTHONPATH=. python3 eucaif/study_arch_residual.py
"""
import math
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
from sage.data.noise import GlitchInjector, scattered_light_arch
from sage.data.noise.glitch_noise import glitch_snr
from sage.dsp.qtransform import QTransform

FS, DUR, SNR = 4096.0, 8.0, 20.0
N = int(FS * DUR)
ASD = math.sqrt(2.0 / FS)
asd = torch.full((N // 2 + 1,), ASD, dtype=torch.float64)
qt = QTransform(FS, N, qrange=(4, 64), frange=(10, 1000), norm=False, dtype=torch.float64)
QS = [float(q) for q in qt.qs]
snr = lambda e: float(glitch_snr(e / ASD, FS))


def setup(seed):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(4, N, generator=g, dtype=torch.float64)
    arch = scattered_light_arch(1, FS, DUR, fringe_frequency=40.0, timeperiod=4.0, dtype=torch.float64,
                                generator=torch.Generator().manual_seed(seed))
    inj = GlitchInjector(FS, f_low=10.0, f_high=1000.0, lowpass_order=None)
    h = inj(torch.zeros(1, 1, N, dtype=torch.float64), arch.view(1, 1, -1), asd, snr=SNR, return_glitch=True)[1][0, 0]
    return noise[0], noise[1:], h


def floor(q, noise):
    return torch.stack([qt.tiles(n, q).abs() ** 2 for n in noise]).mean((0, -1))[:, None]


def binary(q, x, h, noise, kappa=1.0):
    m = (qt.tiles(h, q).abs() ** 2 >= kappa * floor(q, noise)).double()
    return m


def wiener(q, x, h, noise):
    g2, f = qt.tiles(h, q).abs() ** 2, floor(q, noise)
    return g2 / (g2 + f)


def extract(q, x, mask):
    return qt.inverse(qt.tiles(x, q) * mask, q)


rows = {k: [] for k in ("binary Q45", "  noise kept in mask", "  glitch lost outside mask",
                        "Wiener (soft) mask Q45", "best single Q (binary)", "multi-Q pursuit (binary)",
                        "multi-Q pursuit (Wiener)", "template fit (known shape)")}
for seed in range(5):
    n, noise, h = setup(seed)
    x = n + h
    q = 45.3
    m = binary(q, x, h, noise)
    rows["binary Q45"].append(snr(extract(q, x, m) - h))
    rows["  noise kept in mask"].append(snr(extract(q, n, m)))                 # M n
    rows["  glitch lost outside mask"].append(snr(extract(q, h, m) - h))       # M h - h
    rows["Wiener (soft) mask Q45"].append(snr(extract(q, x, wiener(q, x, h, noise)) - h))
    rows["best single Q (binary)"].append(min(snr(extract(p, x, binary(p, x, h, noise)) - h) for p in QS))

    for kind, mk in (("binary", binary), ("Wiener", wiener)):
        # greedy over Q (high to low): each plane extracts what is left of the glitch
        est = torch.zeros(N, dtype=torch.float64)
        for p in sorted(QS, reverse=True):
            left = h - est
            est = est + extract(p, x - est, mk(p, x - est, left, noise))
        rows[f"multi-Q pursuit ({kind})"].append(snr(est - h))

    # parametric (ArchEnemy-style): least-squares amplitude and phase of the arch template
    def fit(f_f, T, t0):
        base = [scattered_light_arch(1, FS, DUR, fringe_frequency=f_f, timeperiod=T, t0=t0, dtype=torch.float64,
                                     generator=torch.Generator().manual_seed(seed))[0]]
        # two quadratures: the template and its Hilbert transform
        B = torch.fft.rfft(base[0])
        quad = torch.fft.irfft(B * -1j, n=N)
        A = torch.stack([base[0], quad], 1)
        A = torch.stack([GlitchInjector(FS, f_low=10.0, f_high=1000.0, lowpass_order=None)
                         (torch.zeros(1, 1, N, dtype=torch.float64), a.view(1, 1, -1), asd, snr=10.0,
                          return_glitch=True)[1][0, 0] for a in A.T], 1)
        coef = torch.linalg.lstsq(A, x[:, None]).solution
        return (A @ coef)[:, 0]

    rows["template fit (known shape)"].append(snr(fit(40.0, 4.0, 4.0) - h))

print(f"scattered light 40 Hz, 4 s, SNR {SNR:g}, white noise; residual SNR of (estimate - injected), 5 seeds")
for k, v in rows.items():
    print(f"{k:28s} mean {np.mean(v):5.2f}  (min {np.min(v):5.2f}, max {np.max(v):5.2f})")


# -- template fit with slightly wrong parameters (glitch alone, no noise) -----------------------
inj = GlitchInjector(FS, f_low=10.0, f_high=1000.0, lowpass_order=None)
col = lambda a: inj(torch.zeros(1, 1, N, dtype=torch.float64), a.view(1, 1, -1), asd, snr=10.0,
                    return_glitch=True)[1][0, 0]
shape = lambda f, T, t0: scattered_light_arch(1, FS, DUR, fringe_frequency=f, timeperiod=T, t0=t0,
                                              dtype=torch.float64, generator=torch.Generator().manual_seed(0))[0]
h = col(shape(40.0, 4.0, 4.0))
h = h * SNR / snr(h)
print("\ntemplate fit with wrong parameters (true: 40 Hz, 4 s, centre 4 s), glitch alone:")
for f, T, t0 in ((40, 4, 4), (40.2, 4, 4), (40.5, 4, 4), (40, 4.05, 4), (40, 4, 4.01)):
    b = col(shape(f, T, t0))
    A = torch.stack([b, torch.fft.irfft(torch.fft.rfft(b) * -1j, n=N)], 1)
    fit = (A @ torch.linalg.lstsq(A, h[:, None]).solution)[:, 0]
    print(f"  f={f:5.1f} Hz T={T:4.2f} s t0={t0:5.2f} s: match {float(fit.norm() / h.norm()):.3f}, "
          f"residual SNR {snr(h - fit):5.2f}")
