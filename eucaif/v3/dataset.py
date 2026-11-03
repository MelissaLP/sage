"""
On-the-fly EuCAIF v3 training data: a torch ``IterableDataset`` that reads real strain from the local v3 cache
(never the network: download it first, ``python -m eucaif.v3.sources download``) and injects BBHs and glitches as
the config asks (``signal.fraction: 0`` and ``glitches: null`` give real noise alone).

    from eucaif.v3.dataset import make_loader
    loader = make_loader("eucaif/v3/eucaif_v3.yaml", split="train", batch_size=64, num_workers=4)
    for epoch in range(10):
        loader.dataset.set_epoch(epoch)
        for b in loader:          # b["x"]: (64, D, L) float32 whitened; b["y"], b["gps"], b["glitch_events"], ...
            ...

Windows are drawn where the cache has data for every detector (a partial download works). With ``split.mode:
random`` every split draws over all the data, as ggwd FM production; with ``gps_blocks`` each split only from its
own GPS blocks. Every batch keeps the GPS time and source of its windows.

Batch ``i`` of epoch ``e`` is fixed by (seed, e, i), whatever the number of workers (each worker makes every
``num_workers``-th batch); its ``recipes`` regenerate any example with ``EucaifV3.replay``. Batches are built whole
by the generator (DataLoader ``batch_size=None``).
"""

import copy

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from eucaif.v3.generator import EucaifV3, load_config


class EucaifV3Stream(IterableDataset):
    """Batches of ``EucaifV3`` from the local cache.

    ``config``: path to the v3 YAML or a loaded config (``generator.load_config``). ``batches``: batches per epoch
    over all workers (None: endless). ``overrides``: {section: {key: value}} applied to the config (e.g.
    ``{"signal": {"fraction": 0}}``).
    """

    def __init__(self, config, split="train", batch_size=64, seed=0, batches=None, overrides=None, device="cpu",
                 log=print):
        cfg = load_config(config) if not isinstance(config, dict) else copy.deepcopy(config)
        for section, values in (overrides or {}).items():
            cfg[section] = values if values is None else dict(cfg.get(section) or {}, **values)
        cfg["cache"] = dict(cfg["cache"], fetch_missing=False)               # training never downloads
        self.cfg, self.split, self.batch_size, self.seed = cfg, split, int(batch_size), int(seed)
        self.batches, self.device, self.log = batches, device, log
        self.epoch = 0

    def set_epoch(self, epoch):
        """Different windows and injections per epoch (call before iterating, as ``DistributedSampler``)."""
        self.epoch = int(epoch)

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        gen = EucaifV3(self.cfg, split=self.split, seed=self.seed, batch_size=self.batch_size, device=self.device,
                       log=self.log if worker == 0 else (lambda *a: None))
        i = worker                                                 # global batch index: same data for any workers
        while self.batches is None or i < self.batches:
            yield gen.batch(key=(self.epoch, i))                   # its recipes replay with EucaifV3.replay
            i += workers


def make_loader(config, split="train", batch_size=64, num_workers=0, seed=0, batches=None, overrides=None,
                device="cpu", **loader_kwargs):
    """A ``DataLoader`` over ``EucaifV3Stream`` (whole batches, so ``batch_size=None`` for the loader)."""
    ds = EucaifV3Stream(config, split=split, batch_size=batch_size, seed=seed, batches=batches,
                        overrides=overrides, device=device)
    return DataLoader(ds, batch_size=None, num_workers=num_workers, **loader_kwargs)
