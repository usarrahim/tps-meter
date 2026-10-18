"""Phase 5 tests: model, loss, sampler, a tiny end-to-end training run with resume."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tpslib.config import load_yaml  # noqa: E402
from tpslib.models.losses import beta_nll, gaussian_nll  # noqa: E402
from tpslib.models.resnet1d import build_model, count_parameters  # noqa: E402
from tpslib.train.trainer import EpochSampler, train  # noqa: E402

CFG = load_yaml("configs/train_m2.yaml")
sys.path.insert(0, str(Path(__file__).parent))
from test_measurement import analytic_case  # noqa: E402


def test_model_shapes_and_size():
    m = build_model(CFG)
    mu, lv = m(torch.randn(5, 3, 256), torch.randn(5, 7))
    assert mu.shape == (5, 3) and lv.shape == (5, 3)
    lo, hi = CFG["model"]["logvar_bounds"]
    assert torch.all(lv > lo) and torch.all(lv < hi)
    assert 300_000 < count_parameters(m) < 450_000      # architecture as specified (DECISIONS.md)


def test_gaussian_nll_minimum_at_true_variance():
    y = torch.zeros(1, 1)
    mu = torch.full((1, 1), 0.3)
    lvs = torch.linspace(-6, 2, 801).view(-1, 1)
    nll = gaussian_nll(mu.expand_as(lvs), lvs, y.expand_as(lvs)).squeeze()
    assert torch.exp(lvs[nll.argmin()]).item() == pytest.approx(0.09, rel=0.02)


def test_beta_nll_weights_and_beta():
    mu, lv, y = torch.zeros(4, 3), torch.zeros(4, 3), torch.ones(4, 3)
    assert beta_nll(mu, lv, y, beta=0.0).item() == pytest.approx(0.5)
    w = beta_nll(mu, lv, y, beta=0.5, weights=[1.0, 0.5, 0.2]).item()
    assert w == pytest.approx(0.5)                     # equal per-target losses -> weights cancel
    lv2 = torch.full((4, 3), 2.0)
    mu.requires_grad_(True)
    beta_nll(mu, lv2, y, beta=0.5).backward()
    g_beta = mu.grad.clone()
    mu.grad = None
    beta_nll(mu, lv2, y, beta=0.0).backward()
    assert torch.allclose(g_beta, mu.grad * np.exp(0.5 * 2.0))   # sg(var^beta) scales the gradient


def test_epoch_sampler_encodes_epoch_and_shuffles():
    s = EpochSampler(10, shuffle=True, seed=3)
    e0 = list(s); s.set_epoch(2); e2 = list(s)
    assert sorted(e0) == list(range(10)) and sorted(e2) == list(range(20, 30))
    assert [x - 20 for x in e2] != e0


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("train")
    ks = np.geomspace(0.1, 20, 24)
    cases = [analytic_case(float(k), n_grid=512) for k in ks]
    for name, sl in (("tr.h5", slice(0, 18)), ("va.h5", slice(18, 24))):
        cs = cases[sl]
        with h5py.File(d / name, "w") as f:
            g = f.create_group("params")
            for key in cs[0][2]:
                g.create_dataset(key, data=np.array([c[2][key] for c in cs], dtype=float))
            f.create_dataset("t", data=np.stack([c[0] for c in cs]))
            f.create_dataset("dT", data=np.stack([c[1] for c in cs]).astype(np.float32))
            f.create_group("diag").create_dataset("ok", data=np.ones(len(cs), int))
    return d


def test_tiny_training_runs_saves_and_resumes(tiny_dataset, tmp_path):
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["train"]["batch_size"] = 6
    cfg["data"]["normalizer_samples"] = 18
    run = tmp_path / "run"
    s1 = train(cfg, tiny_dataset, run, seed=1, train_file="tr.h5", val_file="va.h5", max_epochs=2,
               num_workers=0, log=lambda m: None)
    for f in ("config.yaml", "run_info.json", "normalizer.json", "best.pt", "last.pt", "metrics.csv",
              "learning_curve.png", "summary.json"):
        assert (run / f).exists(), f
    assert s1["epochs_run"] == 2 and np.isfinite(s1["best"]["val_loss"])
    info = json.loads((run / "run_info.json").read_text())
    assert info["n_train"] == 18 and info["n_val"] == 6 and info["code_hash"]
    # resume: continues from epoch 2 instead of starting over
    s2 = train(cfg, tiny_dataset, run, seed=1, train_file="tr.h5", val_file="va.h5", max_epochs=3,
               num_workers=0, log=lambda m: None)
    assert s2["epochs_run"] == 3
    rows = (run / "metrics.csv").read_text().strip().splitlines()
    assert len(rows) == 4


def test_early_stopping_waits_for_min_epochs(tiny_dataset, tmp_path):
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["train"]["batch_size"] = 6
    cfg["train"]["early_stopping"].update(patience=1, min_epochs=4)
    cfg["data"]["normalizer_samples"] = 10
    s = train(cfg, tiny_dataset, tmp_path / "es", seed=3, train_file="tr.h5", val_file="va.h5",
              max_epochs=6, num_workers=0, log=lambda m: None)
    assert s["epochs_run"] >= 5      # never stopped before epoch 4 even with patience 1


def test_contact_asymmetry_filter():
    from tpslib.data.dataset import contact_asymmetry, select_indices
    p = {"k": np.ones(4), "h_c_bottom": np.array([1000.0, 500.0, 50000.0, 2000.0]),
         "h_c_top": np.array([1000.0, 20000.0, 4000.0, 6000.0])}
    assert np.allclose(contact_asymmetry(p), [1.0, 40.0, 12.5, 3.0])
    assert list(select_indices(p, max_contact_asymmetry=10)) == [0, 3]


def test_ablation_band_removed_from_train_and_val(tiny_dataset, tmp_path):
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["train"]["batch_size"] = 6
    cfg["data"].update(normalizer_samples=10, exclude_k_band=[1.0, 5.0])
    s = train(cfg, tiny_dataset, tmp_path / "abl", seed=2, train_file="tr.h5", val_file="va.h5",
              max_epochs=1, num_workers=0, log=lambda m: None)
    ks = np.geomspace(0.1, 20, 24)
    in_band = (ks >= 1.0) & (ks <= 5.0)
    assert s["n_train"] == int((~in_band[:18]).sum()) and s["n_val"] == int((~in_band[18:]).sum())
