#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : __init__.py
Description     : Short description of the file

Created on 2026-01-19 23:28:02

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

from .real_noise import (
    HDF5SingleNoiseSampler,
    MemmapSingleNoiseSampler,
    MemmapNoiseSampler,
)
from .testing_noise import TestNoiseSampler

from .white_noise import (
    sample_synthetic_noise,
    available_asds,
    WhiteGaussianNoiseSampler,
)
from .recolour import RecolourPostprocess
from .glitch_sampler import GlitchOversampledNoiseSampler
from .glitch_noise import GlitchInjector, add_glitch, lowpass_glitch, rescale_glitch, resample_glitch
from .glitch_bank import DeepExtractorGlitchBank, download_deepextractor_bank
from .analytic_glitches import (
    ANALYTIC_GLITCHES,
    analytic_glitch,
    gaussian_blip,
    ringdown,
    sine_gaussian,
    white_noise_burst,
)
from .antiglitch import antiglitch_glitch
from .scattered_light import scattered_light_arch
from .glitch_population import GlitchPopulation, power_law_snr
from .lowfar_noise import StartTimeDataset
from .split import split_observing_run, train_val_split

# NOTE: the hard-negative miner lives in ``sage.data.noise.cma_mae_mining`` and
# is imported directly by the hard-mining trainer (it pulls in pyribs).  It is
# deliberately NOT re-exported here so that importing this package — and the
# default training path — never requires pyribs.

__all__ = [
    "HDF5SingleNoiseSampler",
    "MemmapSingleNoiseSampler",
    "MemmapNoiseSampler",
    "TestNoiseSampler",
    "sample_synthetic_noise",
    "available_asds",
    "WhiteGaussianNoiseSampler",
    "RecolourPostprocess",
    "GlitchOversampledNoiseSampler",
    "GlitchInjector",
    "add_glitch",
    "lowpass_glitch",
    "rescale_glitch",
    "resample_glitch",
    "DeepExtractorGlitchBank",
    "download_deepextractor_bank",
    "ANALYTIC_GLITCHES",
    "analytic_glitch",
    "gaussian_blip",
    "ringdown",
    "sine_gaussian",
    "white_noise_burst",
    "antiglitch_glitch",
    "scattered_light_arch",
    "GlitchPopulation",
    "power_law_snr",
    "StartTimeDataset",
    "split_observing_run",
    "train_val_split",
]
