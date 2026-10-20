"""
Glitch frames for training: generated on the fly, or read from a stored HDF5 file, in the same format.

The hybrid recipe (see ``README_glitch_frames.md``):

- validation / test: fixed sets written once with ``make_glitch_dataset.py`` (different ``--seed``
  each), read with ``StoredGlitchFrames``;
- training: fresh frames every epoch from ``GlitchFramesOnTheFly``, the same generator and config,
  seeded by (seed, epoch, batch), so a run is reproducible.

Both give ``x`` (whitened strain) and ``glitch`` (the whitened glitches alone, the target) over the
stored window ``[q_padding | frame_length | q_padding]``, shape (D, L) per frame.

    train = GlitchFramesOnTheFly("eucaif/sage_analytic_glitches_dense.yaml", batch_size=32,
                                 batches_per_epoch=1000, seed=0)
    loader = DataLoader(train, batch_size=None, num_workers=8)       # items are whole batches
    val = DataLoader(StoredGlitchFrames("val.h5"), batch_size=32, num_workers=4)

    qt = QTransform(fs, L, qrange=(45, 45), frange=(10, 1000), norm=False).to(device)
    cols = frame_columns(qt, 45.0, q_padding, frame_length)          # tile columns inside the frame
    for epoch in range(n_epochs):
        train.set_epoch(epoch)                                          # (with num_workers > 0, use
        for batch in loader:                                            #  persistent_workers=False)
            tiles = qt.tiles(batch["x"].to(device), 45.0)               # (B, D, rows, cols), complex
            out = model(tiles)                                          # e.g. predicted glitch tiles
            pred = qt.inverse(out, 45.0)                                # back to the time domain
            frame = slice(*train.frame_slice)
            loss = ((pred - batch["glitch"].to(device))[..., frame] ** 2).mean()
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_glitch_dataset import GlitchDatasetGenerator   # noqa: E402


def frame_columns(qt, q: float, q_padding: float, frame_length: float) -> slice:
    """Columns of ``qt.tiles(x, q)`` whose times lie in the frame (stored window minus the padding)."""
    t = qt.tile_times(q)
    inside = torch.nonzero((t >= q_padding) & (t < q_padding + frame_length)).flatten()
    return slice(int(inside[0]), int(inside[-1]) + 1)


class GlitchFramesOnTheFly(IterableDataset):
    """Fresh batches of frames, generated from the config; use with ``DataLoader(batch_size=None)``.

    Batch ``i`` of epoch ``e`` is always the same (its seed derives from ``(seed, e, i)``), whatever
    the number of workers: workers share the batches of an epoch round-robin.
    """

    def __init__(self, config, batch_size: int = 32, batches_per_epoch: int = 1000, seed: int = 0,
                 device="cpu", dtype=torch.float32, with_glitch_table: bool = False):
        self.config, self.batch_size, self.batches_per_epoch = str(config), int(batch_size), int(batches_per_epoch)
        self.seed, self.device, self.dtype, self.with_table = int(seed), device, dtype, with_glitch_table
        self.epoch = 0
        self._gen = None
        gen = GlitchDatasetGenerator(self.config, seed=self.seed, device="cpu", dtype=dtype)
        self.frame_slice = (int(round(gen.qpad_s * gen.fs)), int(round((gen.qpad_s + gen.frame_s) * gen.fs)))
        self.t, self.classes, self.detectors = gen.t, gen.classes, gen.detectors

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def batch_seed(self, epoch: int, index: int) -> int:
        """Seed of batch ``index`` of ``epoch`` (disjoint from the small seeds of stored sets)."""
        return int(np.random.SeedSequence([self.seed, epoch, index]).generate_state(1, dtype=np.uint64)[0]) >> 1

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        if self._gen is None:      # built in each worker
            self._gen = GlitchDatasetGenerator(self.config, seed=self.seed, device=self.device, dtype=self.dtype)
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        for i in range(wid, self.batches_per_epoch, nw):
            x, glitch, counts, table = self._gen.generate(self.batch_size, self.batch_seed(self.epoch, i))
            item = {"x": x, "glitch": glitch, "n_glitches": torch.as_tensor(counts.astype(np.int64))}
            if self.with_table:
                item["glitches"] = table
            yield item


class StoredGlitchFrames(Dataset):
    """Frames of a file written by ``make_glitch_dataset.py``: ``x``, ``glitch`` (D, L) per item.

    ``glitches(i)`` returns the rows of the glitch table of frame ``i`` (class, time, SNR, parameters).
    """

    def __init__(self, path, load_table: bool = True):
        import h5py
        self.path = str(path)
        self._f = None
        with h5py.File(self.path, "r") as f:
            self.n = f["x"].shape[0]
            self.t = torch.as_tensor(f["t"][:])
            self.classes = [c.decode() for c in f.attrs["classes"]]
            self.detectors = [d.decode() for d in f.attrs["detectors"]]
            self.frame_slice = tuple(int(v) for v in f.attrs["frame_slice"])
            self.attrs = dict(f.attrs)
            self.table = {k: f[f"glitches/{k}"][:] for k in f["glitches"]} if load_table else None
        if self.table is not None:
            order = np.argsort(self.table["frame"], kind="stable")
            self.table = {k: v[order] for k, v in self.table.items()}
            self._bounds = np.searchsorted(self.table["frame"], np.arange(self.n + 1))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._f is None:        # opened in each worker
            import h5py
            self._f = h5py.File(self.path, "r")
        return {"x": torch.as_tensor(self._f["x"][i]), "glitch": torch.as_tensor(self._f["glitch"][i]),
                "n_glitches": torch.as_tensor(self._f["n_glitches"][i].astype(np.int64)), "index": i}

    def glitches(self, i) -> dict:
        a, b = self._bounds[i], self._bounds[i + 1]
        return {k: v[a:b] for k, v in self.table.items()}
