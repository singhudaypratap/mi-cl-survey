"""Sanity checks against known mathematical identities. Run: python tests/test_core.py"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from micl_pilot import analysis as A  # noqa: E402
from micl_pilot.data import augment  # noqa: E402
from micl_pilot.model import STAGES, CifarResNet18  # noqa: E402
from micl_pilot.train import ReservoirBuffer  # noqa: E402

rng = np.random.default_rng(0)


def test_cka():
    X = rng.normal(size=(300, 20))
    Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    assert abs(A.linear_cka(X, X) - 1) < 1e-12
    # invariant to orthogonal transforms and isotropic scaling
    assert abs(A.linear_cka(X, 3.0 * X @ Q) - 1) < 1e-10
    # independent features -> near 0 (bias O(d/N))
    assert A.linear_cka(X, rng.normal(size=(300, 20))) < 0.15


def test_augment():
    x = torch.randn(5, 3, 32, 32)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(augment(x, 0, False, g), x)
    # crop: every output must equal some 32x32 window of the padded input
    g = torch.Generator().manual_seed(1)
    y = augment(x, 4, False, g)
    xp = F.pad(x, (4, 4, 4, 4))
    for b in range(5):
        hits = [(i, j) for i in range(9) for j in range(9) if torch.equal(xp[b, :, i:i + 32, j:j + 32], y[b])]
        assert len(hits) >= 1


def test_reservoir_uniform():
    counts = np.zeros(1000)
    for trial in range(400):
        buf = ReservoirBuffer(50)
        imgs = torch.zeros((1000, 3, 32, 32), dtype=torch.uint8)
        buf.add(imgs, torch.arange(1000), np.random.default_rng(trial))
        counts[buf.labels.numpy()] += 1
    # each item kept with prob 50/1000; mean count 20 per item, check early vs late halves
    assert abs(counts[:500].mean() - counts[500:].mean()) < 2.0, (counts[:500].mean(), counts[500:].mean())


def test_bases_and_ablation():
    X = np.abs(rng.normal(size=(400, 16)))
    y = np.where(X[:, 0] > X[:, 1], 0, 1)
    probe = A.fit_probe(X, y, {"probe_C": 1.0, "probe_max_iter": 2000})
    neuron = A.Basis("neuron", "neuron", np.zeros(16), np.eye(16), True)
    Acts = neuron.encode(X)
    out = A.ablation_loads(neuron, X, Acts, y, np.array([0, 1]), probe, np.random.default_rng(0))
    for r in ("mean", "resample"):  # complete basis: both spaces identical
        assert out[f"load_{r}_recon"] == out[f"load_{r}_errpres"]
    assert out["load_mean_errpres"] > 0.3  # ablating the two decisive neurons must hurt
    p = A.persistence(Acts, Acts, np.array([0, 3]), 0.5, True)
    assert p["persist_frac_active"] == 1.0 and abs(p["persist_corr"] - 1) < 1e-12
    cfg = {"sae_expansion": 2}
    sae = A.make_sae(X, cfg, 0)
    b = A.sae_basis("r", "random", sae)
    Z = b.encode(X)
    with torch.no_grad():
        ref = (sae.decode_scaled(sae.encode(torch.from_numpy(X).float())) / sae.scale).double().numpy()
    assert np.allclose(b.reconstruct(Z), ref, atol=1e-4)


def test_patching_identity():
    torch.manual_seed(0)
    m = CifarResNet18(10).eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = m.forward_stages(x)
        for s in STAGES:
            assert torch.allclose(m.forward_from(s, out[s]), out["logits"], atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
