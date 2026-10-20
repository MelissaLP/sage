# Noise

## Note: whitening conventions (glitch injection)

The glitch sources and SAGE whiten differently:

- **gengli** (`add_glitch`, its training data) and **DeepExtractor** (training data and the
  glitch bank) whiten with **PyCBC**: Welch PSD (2 s segments, median), inverse spectrum
  truncation (1 s, Hann), then division by the ASD in the frequency domain. The output is
  not normalised, so whitened noise has a large, sample-rate-dependent amplitude (the
  DeepExtractor scaler has a standard deviation of about 284).
- **SAGE** (`sage.dsp.whiten.FIRWhitening`, used for the EuCAIF dataset) reproduces **gwpy**'s
  `TimeSeries.whiten`: a FIR filter (`fduration`, 2 s by default) applied in the time domain,
  with `fduration / 2` dropped at each edge, normalised so whitened Gaussian noise has
  roughly unit variance.

`glitch_noise.py` handles this by colouring each glitch with the target ASD and setting its
amplitude by SNR, not by raw amplitude, then letting SAGE's whitening act on glitch and noise
alike. The SNR doesn't depend on the whitening convention, so the normalisation difference
cancels. What remains is a small difference in glitch shape (different filter lengths and
PSD smoothing, mostly at low frequencies and near strong lines). Keep glitches away from the
edges the whitening discards.

## Glitch sources

All three give whitened glitch shapes for `GlitchInjector` (`glitch_noise.py`), which sets their amplitude by SNR.

| Source | SAGE code | Origin |
|---|---|---|
| gengli (blips) | `glitch_noise.py` (port of `add_glitch`) | [gengli](https://git.ligo.org/melissa.lopez/gengli) |
| Real O3 glitches, 7 Gravity Spy classes, 2 s at 4096 Hz | `glitch_bank.py` (`DeepExtractorGlitchBank`; needs `huggingface_hub`) | [DeepExtractor reconstructions](https://huggingface.co/datasets/tomdooney/deepextractor-glitch-reconstructions), CC BY 4.0, Dooney et al., PRD (2026) |
| Analytic: sine-Gaussian, ringdown, white noise burst, Gaussian blip | `analytic_glitches.py` | [McGANn](https://github.com/jmcginn/McGANn) `cGAN/GAN_training_data.py`, MIT, McGinn et al., [CQG 38 155005 (2021)](https://doi.org/10.1088/1361-6382/ac09cc) |

The differences from each original are listed in the module docstrings.
