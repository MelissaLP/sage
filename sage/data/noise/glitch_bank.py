#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : glitch_bank.py
Description     : The DeepExtractor glitch bank (real O3 glitches) from Hugging Face.

GitHub Repository: NULL

Documentation:

    Source: https://huggingface.co/datasets/tomdooney/deepextractor-glitch-reconstructions
    (CC BY 4.0). DeepExtractor (https://github.com/tomdooney95/deepextractor) reconstructions
    of O3a/O3b H1 and L1 glitches with Gravity Spy confidence >= 0.9 and SNR >= 15: 2 s at
    4096 Hz (8192 samples), whitened, mean-subtracted and normalised to [-1, 1]. Cite:

        T. Dooney et al., "Realistic Time-Domain Synthesis of Gravitational-Wave Detector
        Glitches using Class-Conditional Derivative Generative Adversarial Networks",
        Phys. Rev. D (2026).

    Only the three files needed for injection are downloaded (~2.3 GB): the waveforms, their
    one-hot labels and the class names. The derivatives (another 2.3 GB, for GAN training),
    the GlitchGAN model and the hold-out set are not. Files go to the Hugging Face cache
    (``HF_HOME``, set per server by ``sage/utils/run_base.sh``) or to ``data_dir``, at the
    pinned ``HF_REVISION`` so a dataset can be traced to the exact glitches it used.

        bank = DeepExtractorGlitchBank()                         # downloads once, then cached
        train, test = bank.split(0.8, seed=0)                    # disjoint unique glitches
        glitches, labels = bank.sample(64, classes=["Blip"], indices=train, sample_rate=2048)
        strain = GlitchInjector(2048, f_low=20.0)(strain, glitches.view(B, D, -1), asd, snr)

    The bank was class-balanced to 5000 per class by bootstrap resampling, so rare classes
    repeat glitches. ``deduplicate=True`` (default) keeps the first copy of each, so a
    train/test split cannot put the same glitch on both sides.

    The waveforms are memory-mapped: only the glitches drawn are read from disk.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .glitch_noise import resample_glitch

HF_REPO_ID = "tomdooney/deepextractor-glitch-reconstructions"
HF_REVISION = "144f56880c6e7aa8def31c537ca843b8c9e5bdda"   # main on 2026-08-25
SAMPLE_RATE = 4096.0
FILES = {
    "samples": "glitch_GAN_samples_scaled_balanced.npy",   # (35000, 8192) float64
    "labels": "glitch_GAN_labels_balanced.npy",            # (35000, 7) one-hot
    "label_order": "glitch_GAN_label_order.npy",           # (7,) class names
}


def download_deepextractor_bank(cache_dir=None, local_dir=None, revision: str = HF_REVISION) -> dict:
    """Download the bank's waveforms, labels and class names; return their local paths.

    Needs ``huggingface_hub`` (``pip install huggingface_hub``). Files already in the cache
    (or in ``local_dir``) are not downloaded again.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("Downloading the DeepExtractor glitch bank needs huggingface_hub: "
                          "pip install huggingface_hub") from e
    return {key: Path(hf_hub_download(repo_id=HF_REPO_ID, filename=name, repo_type="dataset",
                                      revision=revision, cache_dir=cache_dir, local_dir=local_dir))
            for key, name in FILES.items()}


def _unique_rows(x: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Indices of the first occurrence of each distinct row, in order (reads ``x`` in chunks)."""
    seen, keep = set(), []
    for start in range(0, len(x), chunk):
        block = np.ascontiguousarray(x[start:start + chunk])
        for i, row in enumerate(block):
            digest = hashlib.blake2b(row.tobytes(), digest_size=16).digest()
            if digest not in seen:
                seen.add(digest)
                keep.append(start + i)
    return np.asarray(keep, dtype=np.int64)


class DeepExtractorGlitchBank:
    """Real O3 glitches (7 Gravity Spy classes) reconstructed by DeepExtractor.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory holding the bank's files. Missing files are downloaded into it (if
        ``download``). Default: the Hugging Face cache.
    download : bool
        Download missing files. False raises if they are not already there.
    revision : str
        Dataset revision (commit hash) to download.
    cache_dir : str or Path, optional
        Hugging Face cache directory when ``data_dir`` is None (default: ``HF_HOME``).
    deduplicate : bool
        Drop the bootstrap duplicates, keeping the first copy of each glitch. Reads the
        whole bank once (a few seconds for the full 2.3 GB).

    Attributes
    ----------
    classes : tuple of str
        Class names, in label order: Blip, Fast_Scattering, Koi_Fish, Low_Frequency_Burst,
        Scattered_Light, Tomte, Whistle.
    indices : ndarray
        Bank rows in use (all rows, or the unique ones).
    labels : ndarray
        Class index of every bank row.
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, data_dir=None, download: bool = True, revision: str = HF_REVISION,
                 cache_dir=None, deduplicate: bool = True):
        if data_dir is not None:
            paths = {key: Path(data_dir) / name for key, name in FILES.items()}
            if not all(p.exists() for p in paths.values()):
                if not download:
                    missing = [str(p) for p in paths.values() if not p.exists()]
                    raise FileNotFoundError(f"Glitch bank files missing: {missing}")
                paths = download_deepextractor_bank(local_dir=data_dir, revision=revision)
        elif download:
            paths = download_deepextractor_bank(cache_dir=cache_dir, revision=revision)
        else:
            raise ValueError("Give data_dir, or allow download")

        self.samples = np.load(paths["samples"], mmap_mode="r")
        self.labels = np.argmax(np.load(paths["labels"]), axis=1)
        self.classes = tuple(str(c) for c in np.load(paths["label_order"]))
        if len(self.labels) != len(self.samples):
            raise ValueError(f"{len(self.samples)} waveforms but {len(self.labels)} labels")
        self.indices = _unique_rows(self.samples) if deduplicate else np.arange(len(self.samples))

    def __len__(self) -> int:
        return len(self.indices)

    def class_indices(self, classes=None, indices=None) -> np.ndarray:
        """Bank rows (from ``indices``, default all in use) of the given class names or indices."""
        indices = self.indices if indices is None else np.asarray(indices)
        if classes is None:
            return indices
        if isinstance(classes, (str, int)):
            classes = [classes]
        wanted = [self.classes.index(c) if isinstance(c, str) else int(c) for c in classes]
        return indices[np.isin(self.labels[indices], wanted)]

    def split(self, fraction: float, seed=None):
        """Split the rows in use into two disjoint sets, ``fraction`` of each class in the first."""
        rng = np.random.default_rng(seed)
        first, second = [], []
        for c in range(len(self.classes)):
            rows = rng.permutation(self.class_indices(c))
            k = int(round(fraction * len(rows)))
            first.append(rows[:k])
            second.append(rows[k:])
        return np.sort(np.concatenate(first)), np.sort(np.concatenate(second))

    def get(self, rows, sample_rate: float = None, dtype=torch.float32, device=None) -> Tensor:
        """Glitches at bank ``rows``, shape (n, L), resampled to ``sample_rate`` if given."""
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows)                         # read the memmap in order
        data = np.empty((len(rows), self.samples.shape[1]), dtype=np.float64)
        data[order] = self.samples[rows[order]]
        glitch = torch.from_numpy(data)
        if sample_rate is not None:
            glitch = resample_glitch(glitch, self.sample_rate, sample_rate)
        return glitch.to(dtype=dtype, device=device)

    def sample(self, n: int, classes=None, indices=None, generator: torch.Generator = None,
               sample_rate: float = None, dtype=torch.float32, device=None):
        """Draw ``n`` glitches (with replacement) from ``classes`` among ``indices``.

        Returns ``(glitches, labels)``: shape (n, L) and the class index of each glitch.
        """
        pool = self.class_indices(classes, indices)
        if len(pool) == 0:
            raise ValueError(f"No glitches of classes {classes} in the given indices")
        pick = torch.randint(len(pool), (n,), generator=generator).numpy()
        rows = pool[pick]
        return (self.get(rows, sample_rate=sample_rate, dtype=dtype, device=device),
                torch.as_tensor(self.labels[rows], device=device))
