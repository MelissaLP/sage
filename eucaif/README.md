# EuCAIF datasets and studies

Code, configs and documentation of the EuCAIF datasets built with SAGE. Run everything from the SAGE repo root
with `PYTHONPATH=.`, in the environment of `eucaif/environment.yml`.

| Folder | What |
|---|---|
| [`v1/`](v1/README.md) | v1: Gaussian noise + BBH (IMRPhenomPv2, physical distances), 2 s windows, HDF5 files |
| [`v2/`](v2/README.md) | v2: v1 + O3a-calibrated analytic and McGAN glitches, 4 s stored windows, on the fly or HDF5 |
| [`v3/`](v3/README.md) | v3: real H1/L1 strain (GWOSC or frames) + BBH and glitch injections, on the fly from a local cache |
| [`glitch_frames/`](glitch_frames/README.md) | frames dense with analytic glitches, for glitch extraction (DeepExtractor-style) |
| [`benchmark/`](benchmark/README.md) | Q-transform benchmark (speed and gwpy agreement) and the glitch-extraction comparisons |
| `examples/` | short examples: glitch models, noise Q-transform, scattered light, each version's model inputs |
| [`vigilant/`](vigilant/README.md) | whitened Virgo glitches for the Vigilant project |
| `notes/` | LaTeX notes (centred crop, comparison with ggwd production) |

`inspect_dataset.py` summarises and checks the HDF5 files of v1, v2 and the glitch frames:
`python eucaif/inspect_dataset.py check FILE`.

## Where things moved

Before the restructure everything sat at the top of `eucaif/`. Old paths and imports, and their new ones:

| Before | Now |
|---|---|
| `README_dataset.md`, `sage_eucaif_waveform.yaml`, `make_whitened_examples.py` | `v1/` (`README.md`) |
| `README_eucaif_v2.md`, `eucaif_v2.yaml`, `eucaif_v2.py`, `example_eucaif_v2.py`, `sage_analytic_glitches.yaml`, `gravityspy_o3a_rates.py` | `v2/` (`README.md`) |
| `eucaif_v3.yaml`, `example_eucaif_v3.py` | `v3/` |
| `README_glitch_frames.md`, `glitch_frames.py`, `make_glitch_dataset.py`, `sage_analytic_glitches_dense.yaml`, `example_dense_glitches.py` | `glitch_frames/` (`README.md`) |
| `README_qtransform_benchmark.md`, `benchmark_qtransform.py`, `compare_*.py`, `residual_snr_by_class.py`, `study_arch_residual.py`, `test_deepextractor_arch.py`, `benchmark_antiglitch.py`, `example_glitch_transforms.py` | `benchmark/` (`README.md`) |
| `example_glitch_models.py`, `example_noise_qtransform.py`, `example_scattered_light.py`, `example_model_inputs.py`, `testing_sage.ipynb` | `examples/` |
| `figures/` | each folder's own `figures/` |
| `from eucaif.eucaif_v2 import ...` | `from eucaif.v2.eucaif_v2 import ...` |
| `from eucaif.make_whitened_examples import ...` | `from eucaif.v1.make_whitened_examples import ...` |
| `from eucaif.glitch_frames import ...` | `from eucaif.glitch_frames.glitch_frames import ...` |
| `from eucaif.make_glitch_dataset import ...` | `from eucaif.glitch_frames.make_glitch_dataset import ...` |

Tagged releases (`eucaif-dataset-v1.0` to `v3.0`, `eucaif-glitch-frames-v1.0` to `v1.2`) keep the old layout: to
reproduce one of their files, check out its tag and follow the README of that tag.
