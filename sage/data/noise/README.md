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
