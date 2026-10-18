"""PyTorch dataset: clean curves from HDF5 -> measurement effects -> preprocess -> features.

- Training: measurement effects are drawn anew every epoch (augmentation); seed = (base, epoch, index).
- Evaluation (val/test): effects are fixed per sample (epoch ignored), so metrics are reproducible.
- The HDF5 file is opened lazily inside each DataLoader worker (Windows spawn friendly).
- Normalisation statistics are fitted on the TRAIN split only and saved with the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from tpslib.config import CONFIG_DIR, load_yaml
from tpslib.data.features import CHANNELS, SCALARS, TARGETS, make_features, targets
from tpslib.measurement.acquisition import acquire
from tpslib.measurement.preprocess import preprocess

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch is a project dependency
    torch, Dataset = None, object


def load_params(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {k: f["params"][k][()] for k in f["params"]}


def contact_asymmetry(params: dict) -> np.ndarray:
    """h_c,max / h_c,min of the two sensor faces."""
    hb, ht = params["h_c_bottom"], params["h_c_top"]
    return np.maximum(hb, ht) / np.minimum(hb, ht)


def select_indices(params: dict, exclude_k_band=None, only_ok: np.ndarray | None = None,
                   max_contact_asymmetry: float | None = None) -> np.ndarray:
    keep = np.ones(params["k"].shape[0], bool)
    if max_contact_asymmetry is not None:
        keep &= contact_asymmetry(params) <= float(max_contact_asymmetry)
    if exclude_k_band is not None:
        lo, hi = exclude_k_band
        keep &= ~((params["k"] >= lo) & (params["k"] <= hi))
    if only_ok is not None:
        keep &= only_ok.astype(bool)
    return np.where(keep)[0]


class TPSDataset(Dataset):
    def __init__(self, path, train: bool, base_seed: int, indices=None, normalizer=None, effects=True):
        self.path = Path(path)
        self.train = train
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.normalizer = normalizer
        self.effects = effects
        self.params = load_params(self.path)
        with h5py.File(self.path, "r") as f:
            ok = f["diag/ok"][()] if "diag" in f else np.ones(self.params["k"].shape[0])
        self.indices = select_indices(self.params, only_ok=ok) if indices is None else np.asarray(indices)
        self._h5 = None
        self._device = load_yaml(CONFIG_DIR / "device.yaml")
        self._sensor = load_yaml(CONFIG_DIR / "sensor.yaml")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.indices.size)

    def row(self, i: int) -> dict:
        return {k: v[i].item() for k, v in self.params.items()}

    def processed(self, j: int):
        """Preprocessed record (t, dT, P) of item j exactly as `features` sees it (for M0)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        epoch, j = divmod(int(j), len(self))
        epoch = (epoch if epoch > 0 else self.epoch) if self.train else 0
        i = int(self.indices[j])
        prm = self.row(i)
        rng = np.random.default_rng([self.base_seed, epoch, i])
        t, dT = self._h5["t"][i], self._h5["dT"][i].astype(np.float64)
        return preprocess(acquire(t, dT, prm, rng, self._device, self._sensor, effects=self.effects)), prm

    def features(self, j: int) -> tuple[dict, np.ndarray]:
        """Features and targets of dataset item j (index into self.indices).

        j >= len(self) encodes the epoch: j = epoch * len + item (see train.trainer.EpochSampler), so
        persistent DataLoader workers draw new measurement effects every epoch without shared state.
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        epoch, j = divmod(int(j), len(self))
        epoch = epoch if epoch > 0 else self.epoch
        i = int(self.indices[j])
        prm = self.row(i)
        epoch = epoch if self.train else 0
        rng = np.random.default_rng([self.base_seed, epoch, i])
        t, dT = self._h5["t"][i], self._h5["dT"][i].astype(np.float64)
        rec = acquire(t, dT, prm, rng, self._device, self._sensor, effects=self.effects)
        return make_features(preprocess(rec)), targets(prm)

    def __getitem__(self, j: int):
        feat, y = self.features(j)
        seq, scal = feat["seq"], feat["scalars"]
        if self.normalizer is not None:
            seq, scal, y = self.normalizer.apply(seq, scal, y)
        return torch.from_numpy(seq), torch.from_numpy(scal), torch.from_numpy(y)


@dataclass
class Normalizer:
    """Per-channel (sequence), per-feature (scalars) and per-target mean/std, fitted on train only."""

    seq_mean: np.ndarray
    seq_std: np.ndarray
    scal_mean: np.ndarray
    scal_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    fitted_on: str = ""
    n_samples: int = 0

    @classmethod
    def fit(cls, ds: TPSDataset, n: int = 5000, seed: int = 0) -> "Normalizer":
        if not ds.train:
            raise ValueError("normalisation statistics must come from the training split")
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
        seqs, scals, ys = [], [], []
        for j in pick:
            f, y = ds.features(int(j))
            seqs.append(f["seq"]); scals.append(f["scalars"]); ys.append(y)
        S, C, Y = np.stack(seqs), np.stack(scals), np.stack(ys)
        sd = lambda x: np.where(x > 1e-12, x, 1.0)
        return cls(seq_mean=S.mean(axis=(0, 2)), seq_std=sd(S.std(axis=(0, 2))),
                   scal_mean=C.mean(axis=0), scal_std=sd(C.std(axis=0)),
                   y_mean=Y.mean(axis=0), y_std=sd(Y.std(axis=0)),
                   fitted_on=str(ds.path.name), n_samples=int(pick.size))

    def apply(self, seq, scal, y):
        seq = (seq - self.seq_mean[:, None]) / self.seq_std[:, None]
        scal = (scal - self.scal_mean) / self.scal_std
        y = (y - self.y_mean) / self.y_std
        return seq.astype(np.float32), scal.astype(np.float32), y.astype(np.float32)

    def save(self, path) -> None:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        d.update({"channels": CHANNELS, "scalars": SCALARS, "targets": TARGETS})
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Normalizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        for k in ("channels", "scalars", "targets"):
            d.pop(k, None)
        arr = {k: np.asarray(v, dtype=np.float64) if isinstance(v, list) else v for k, v in d.items()}
        return cls(**arr)
