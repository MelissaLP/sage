#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : gengli_glitches.py
Description     : gengli's GAN blips as torch tensors, with the interface of the analytic glitches.

GitHub Repository: NULL

Documentation:

    gengli (https://git.ligo.org/melissa.lopez/gengli, docs https://gengli-ed413e.docs.ligo.org/,
    GPL-3.0, ``pip install gengli``) generates whitened blip glitches with a GAN trained on
    Gravity Spy blips of LIGO Hanford (H1) or Livingston (L1). Cite:

        M. Lopez et al., "Simulating transient noise bursts in LIGO with generative adversarial
        networks", Phys. Rev. D 106, 023027 (2022), arXiv:2203.06494;
        M. Lopez et al., "Simulating Transient Noise Bursts in LIGO with gengli", arXiv:2205.09204.

    ``gengli_glitch`` takes the same arguments as the generators of ``analytic_glitches``
    (``n``, ``sample_rate``, ``duration``, ``t0``, ``generator``, ``dtype``, ``device``,
    ``return_params``), so ``GlitchPopulation`` configs can use ``generator: gengli_glitch``,
    and its output goes to ``GlitchInjector``, which sets the SNR:

        blip = gengli_glitch(B * D, sample_rate=4096, t0=0.5).view(B, D, -1)
        strain = GlitchInjector(4096, f_low=20.0)(strain, blip, asd, snr, t_inj=t_inj)

    Each blip is gengli's ``get_glitch`` (resampled to ``sample_rate``, optionally low-passed at
    ``fhigh``, Tukey-tapered with ``alpha``), scaled to a peak |h| of 1 and centred at ``t0`` in
    an array of ``duration`` seconds. A blip lasts about 0.23 s.

    gengli seeds torch's global generator; here its seed is drawn from ``generator`` and the
    global state is restored afterwards, so the draws are reproducible and leave the caller's
    random state alone. gengli is imported on first use: SAGE does not require it.
"""

import functools

import numpy as np
import torch

from .analytic_glitches import _peak_normalise, _uniform
from .glitch_noise import place_glitch


@functools.lru_cache(maxsize=None)
def _gengli_generator(detector: str):
    import gengli
    return gengli.glitch_generator(detector)


def gengli_glitch(n: int, sample_rate: float, duration: float = 1.0, t0=None, detector: str = "L1",
                  fhigh: float = None, alpha: float = 0.25, generator: torch.Generator = None,
                  dtype=torch.float32, device=None, return_params: bool = False):
    """``n`` gengli blips, shape (n, duration * sample_rate).

    Parameters
    ----------
    t0 : (low, high), float or tensor, optional
        Centre of each blip, in seconds from the start. Default: ``duration / 2``.
    detector : str
        "H1" or "L1": which of gengli's trained GANs to use.
    fhigh : float, optional
        gengli's low-pass cutoff in Hz (it suggests 250 Hz for blips). None for no filter.
    alpha : float
        gengli's Tukey taper of each blip (its default, 0.25). None or 0 for none.
    """
    work = torch.float64
    t0 = _uniform(duration / 2 if t0 is None else t0, n, generator, work)
    seed = int(torch.randint(2 ** 31 - 1, (1,), generator=generator))
    with torch.random.fork_rng():
        blip = _gengli_generator(detector).get_glitch(n_glitches=n, srate=float(sample_rate), seed=seed,
                                                      alpha=alpha, fhigh=fhigh)
    # gengli squeezes its output: one glitch comes back as (L,), not (1, L)
    blip = torch.from_numpy(np.atleast_2d(blip).astype(np.float64)).to(device)
    n_out = int(round(duration * sample_rate))
    start = torch.round(t0 * sample_rate).long() - blip.shape[-1] // 2
    h = place_glitch(_peak_normalise(blip), n_out, start.to(device)).to(dtype)
    if return_params:
        return h, dict(t0=t0.to(device))
    return h
