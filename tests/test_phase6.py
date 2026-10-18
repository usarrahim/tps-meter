"""Phase 6 tests: shared-feature ensemble prediction, trust thresholds, ONNX export."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from test_train import CFG, tiny_dataset  # noqa: E402,F401  (fixture)

from tpslib.data.dataset import TPSDataset  # noqa: E402
from tpslib.eval.predict import load_run, predict, predict_members  # noqa: E402
from tpslib.eval.trust import fit_thresholds, flags  # noqa: E402
from tpslib.models.export import EnsembleWithNorm, export_onnx, verify_onnx  # noqa: E402
from tpslib.train.trainer import train  # noqa: E402


@pytest.fixture(scope="module")
def two_runs(tiny_dataset, tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    cfg = copy.deepcopy(CFG)
    cfg["train"]["batch_size"] = 6
    cfg["data"]["normalizer_samples"] = 18
    for s in (1, 2):
        train(cfg, tiny_dataset, d / f"r{s}", seed=s, train_file="tr.h5", val_file="va.h5", max_epochs=1,
              num_workers=0, log=lambda m: None)
    return [load_run(d / f"r{s}", torch.device("cpu")) for s in (1, 2)]


def test_predict_members_matches_predict(two_runs, tiny_dataset):
    ds = TPSDataset(tiny_dataset / "va.h5", train=False, base_seed=2002)
    many = predict_members([(m, nm) for m, nm, _ in two_runs], ds)
    for (m, nm, _), p in zip(two_runs, many):
        one = predict(m, nm, TPSDataset(tiny_dataset / "va.h5", train=False, base_seed=2002))
        assert np.allclose(one["mean"], p["mean"], atol=1e-5) and np.allclose(one["std"], p["std"], rtol=1e-5)
        assert np.allclose(one["true"], p["true"], atol=1e-6)


def test_onnx_ensemble_equals_pytorch(two_runs, tiny_dataset, tmp_path):
    pytest.importorskip("onnxruntime")
    em = EnsembleWithNorm([m for m, _, _ in two_runs], [nm for _, nm, _ in two_runs])
    path = export_onnx(em, tmp_path / "e.onnx", 3, 7)
    ds = TPSDataset(tiny_dataset / "va.h5", train=False, base_seed=2002)
    feats = [ds.features(j)[0] for j in range(len(ds))]
    seq = np.stack([f["seq"] for f in feats]).astype(np.float32)
    sc = np.stack([f["scalars"] for f in feats]).astype(np.float32)
    d = verify_onnx(em, path, seq, sc)
    assert max(d.values()) < 1e-4
    # graph output equals the ensemble of the individual PyTorch predictions
    from tpslib.eval.predict import ensemble
    ref = ensemble(predict_members([(m, nm) for m, nm, _ in two_runs], ds))
    with torch.no_grad():
        mu, sd, _ = em(torch.from_numpy(seq), torch.from_numpy(sc))
    assert np.allclose(mu.numpy(), ref["mean"], atol=1e-4) and np.allclose(sd.numpy(), ref["std"], rtol=1e-3, atol=1e-5)


def test_trust_thresholds_false_alarm_rate():
    rng = np.random.default_rng(0)
    mk = lambda n, shift=0.0: {"epistemic": rng.exponential(1, n) + shift, "total_std": rng.exponential(1, n),
                               "m0_m2": rng.exponential(1, n), "m0_resid": rng.exponential(1, n),
                               "dT_max": np.full(n, 2.5)}
    thr = fit_thresholds(mk(20000), q=0.99)
    normal = flags(mk(20000), thr)
    assert 0.02 < normal["any"].mean() < 0.06          # 4 checks x ~1 % each
    bad = flags(mk(2000, shift=10.0), thr)
    assert bad["epistemic"].mean() > 0.99
    nan = mk(10); nan["m0_m2"][:] = np.nan
    assert flags(nan, thr)["m0_m2"].all()              # failed classic fit is flagged
