"""Mechanistic measurements at each checkpoint (protocol Sec. sec:protocol, vision testbed).

Conventions
-----------
* Task indices are 0-based in code; "task 1" / "checkpoint 1" in the paper is index 0 here.
* t0 = the checkpoint after the first task. All dictionaries, unit selections and frozen probes
  are fitted at t0 (or, for frozen probes of task j, at the checkpoint after task j) and then
  never refitted.
* Penultimate features ("pen") are the 512-d global-average-pooled output of layer4.
  The mid-layer features ("mid") are the 256-d global-average-pooled output of layer3.
* Every unit basis is written as x = offset + A @ D.T + err, with A (N, m) activations and
  D (d, m) unit directions in the original feature units. This lets the SAE, the random
  dictionary, the neuron basis and the PCA basis share one persistence/ablation code path.
"""
import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import augment, normalize
from .model import STAGES

log = logging.getLogger("micl")


# --------------------------------------------------------------------------- features
@torch.no_grad()
def extract(model, split, cfg, device, aug_generator):
    """Pooled features and logits in fp32 (no AMP for analysis). Returns float64 numpy."""
    model.eval()
    pen, mid, logits = [], [], []
    bs = cfg["eval_batch_size"]
    for i in range(0, len(split.labels), bs):
        x = normalize(split.images[i:i + bs].to(device), cfg["norm_mean"], cfg["norm_std"])
        if aug_generator is not None:
            x = augment(x, cfg["aug_crop_padding"], cfg["aug_hflip"], aug_generator)
        out = model.forward_stages(x)
        pen.append(out["pen"].cpu())
        mid.append(model.pool(out[cfg["mid_layer"]]).cpu())
        logits.append(out["logits"].cpu())
    to = lambda xs: torch.cat(xs).double().numpy()
    return {"pen": to(pen), "mid": to(mid), "logits": to(logits), "y": split.labels.numpy().copy()}


def output_accuracy(logits, y, allowed_classes):
    """Argmax restricted to `allowed_classes` (CIL: all trained classes; TIL: the task's)."""
    allowed = np.asarray(allowed_classes)
    pred = allowed[np.argmax(logits[:, allowed], axis=1)]
    return float(np.mean(pred == y))


# --------------------------------------------------------------------------- probes
def fit_probe(X, y, cfg):
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(C=cfg["probe_C"], max_iter=cfg["probe_max_iter"]))
    probe.fit(X, y)
    return probe


def probe_accuracy(probe, X, y):
    return float(np.mean(probe.predict(X) == y))


# --------------------------------------------------------------------------- CKA / capacity
def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019) between row-aligned feature matrices."""
    assert X.shape[0] == Y.shape[0]
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    assert denom > 0
    return float(hsic / denom)


def capacity_stats(pen):
    """Dead-unit fraction (never active on the eval set) and effective rank
    (exp of the entropy of normalised singular values; Roy & Vetterli 2007)."""
    dead = float(np.mean(pen.max(0) <= 0.0))
    s = np.linalg.svd(pen - pen.mean(0, keepdims=True), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return {"dead_unit_frac": dead, "effective_rank": float(np.exp(-(p * np.log(p)).sum()))}


# --------------------------------------------------------------------------- dictionaries
class SAE(torch.nn.Module):
    """ReLU sparse autoencoder. Inputs are multiplied by a fixed scalar `scale` so that
    E||x * scale||^2 = d on the t0 training features; decoder columns have unit norm."""

    def __init__(self, d, m, scale, b_dec_init, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w_dec = torch.randn(d, m, generator=g)
        w_dec = w_dec / w_dec.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(w_dec)
        self.W_enc = torch.nn.Parameter(w_dec.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.b_dec = torch.nn.Parameter(b_dec_init.clone().float())
        self.register_buffer("scale", torch.tensor(float(scale)))

    def encode(self, x):
        return torch.relu((x * self.scale - self.b_dec) @ self.W_enc.T + self.b_enc)

    def decode_scaled(self, z):
        return z @ self.W_dec.T + self.b_dec

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True)


def make_sae(X_train, cfg, seed):
    d = X_train.shape[1]
    m = cfg["sae_expansion"] * d
    scale = np.sqrt(d / np.mean(np.sum(X_train ** 2, axis=1)))
    b_dec = torch.from_numpy(X_train.mean(0) * scale)
    return SAE(d, m, scale, b_dec, seed)


def train_sae(X_train, cfg, seed, device):
    """Train on t0 features. Loss = mean_n ||xs - x̂s||^2 / d + l1 * mean_n sum_j z_j ||W_dec_j||."""
    sae = make_sae(X_train, cfg, seed).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg["sae_lr"])
    X = torch.from_numpy(X_train).float().to(device)
    d = X.shape[1]
    g = torch.Generator().manual_seed(seed + 31)
    for epoch in range(cfg["sae_epochs"]):
        perm = torch.randperm(X.shape[0], generator=g)
        tot_mse, tot_l1, nb = 0.0, 0.0, 0
        for i in range(0, X.shape[0], cfg["sae_batch_size"]):
            xb = X[perm[i:i + cfg["sae_batch_size"]].to(device)]
            z = sae.encode(xb)
            xs_hat = sae.decode_scaled(z)
            mse = ((xb * sae.scale - xs_hat) ** 2).sum(1).mean() / d
            l1 = (z * sae.W_dec.norm(dim=0)).sum(1).mean()
            loss = mse + cfg["sae_l1"] * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            tot_mse += mse.item(); tot_l1 += l1.item(); nb += 1
        if epoch == cfg["sae_epochs"] - 1 or epoch % 10 == 0:
            log.info("SAE seed %d epoch %d/%d mse/d %.4f l1 %.3f", seed, epoch + 1,
                     cfg["sae_epochs"], tot_mse / nb, tot_l1 / nb)
    return sae.cpu().eval()


@dataclass
class Basis:
    name: str
    kind: str              # "sae", "random", "neuron", "pca"
    offset: np.ndarray     # (d,)
    D: np.ndarray          # (d, m) unit directions in original feature units
    nonneg: bool           # activations are >= 0 (fraction-active persistence defined)
    sae: object = None     # SAE module for sae/random, None otherwise
    pca: object = None

    def encode(self, X):
        if self.kind in ("sae", "random"):
            with torch.no_grad():
                return self.sae.encode(torch.from_numpy(X).float()).double().numpy()
        if self.kind == "neuron":
            return X.copy()
        if self.kind == "pca":
            return (X - self.offset) @ self.D
        raise ValueError(self.kind)

    def reconstruct(self, A):
        return self.offset + A @ self.D.T


def sae_basis(name, kind, sae):
    s = float(sae.scale)
    return Basis(name, kind, offset=sae.b_dec.detach().double().numpy() / s,
                 D=sae.W_dec.detach().double().numpy() / s, nonneg=True, sae=sae)


def build_bases(X_train_t0, cfg, device):
    """All unit bases, fitted (or drawn) once at t0 and frozen afterwards."""
    d = X_train_t0.shape[1]
    bases, sae_stats = [], {}
    for s in cfg["sae_seeds"]:
        sae = train_sae(X_train_t0, cfg, s, device)
        bases.append(sae_basis(f"sae_s{s}", "sae", sae))
        # Random-dictionary control: identical construction, same size, never trained.
        bases.append(sae_basis(f"random_s{s}", "random", make_sae(X_train_t0, cfg, 10_000 + s).eval()))
    bases.append(Basis("neuron", "neuron", offset=np.zeros(d), D=np.eye(d), nonneg=True))
    # PCA basis of matched *input* size: d components (a complete basis; it cannot be 8x
    # overcomplete). Only in the smoke test (N < d) is it truncated to N components.
    n_pc = min(d, X_train_t0.shape[0])
    if n_pc < d:
        log.warning("PCA basis truncated to %d < d=%d components (too few samples)", n_pc, d)
    pca = PCA(n_components=n_pc, svd_solver="full").fit(X_train_t0)
    bases.append(Basis("pca", "pca", offset=pca.mean_.copy(), D=pca.components_.T.copy(), nonneg=False, pca=pca))
    for b in bases:
        sae_stats[b.name] = dictionary_stats(b, X_train_t0)
        log.info("basis %s at t0 (train): %s", b.name, sae_stats[b.name])
    return bases, sae_stats


def explained_variance(X, X_hat):
    return float(1.0 - np.sum((X - X_hat) ** 2) / np.sum((X - X.mean(0, keepdims=True)) ** 2))


def dictionary_stats(basis, X):
    A = basis.encode(X)
    return {"explained_variance": explained_variance(X, basis.reconstruct(A)),
            "l0": float(np.mean(np.sum(A > 0, axis=1))) if basis.nonneg else float("nan"),
            "dead_frac": float(np.mean(A.max(0) <= 0)) if basis.nonneg else float("nan"),
            "n_units": int(A.shape[1])}


# --------------------------------------------------------------------------- unit selection
def logit_diff_directions(W_fc, y, task_cls):
    """Per-sample direction w_n = W_fc[y_n] - W_fc[other task class], shape (N, d)."""
    assert len(task_cls) == 2, "attribution is defined for 2-class tasks"
    other = np.where(y == task_cls[0], task_cls[1], task_cls[0])
    return W_fc[y] - W_fc[other]


def select_units(basis, X, y, W_fc, task_cls, k, min_fire_rate=0.01):
    """Top-k units by mean attribution to the task-1 correct-vs-other logit difference at t0,
    attribution_nj = A_nj * (w_n . D_j). Selected on t0 *training* features; evaluated on test.
    For non-negative dictionaries, only units that fire on at least `min_fire_rate` of the
    task-1 training inputs are eligible, so ultra-sparse latents (which may never fire on the
    test split) are not selected. Selection never looks at test data."""
    A = basis.encode(X)
    W = logit_diff_directions(W_fc, y, task_cls)  # (N, d)
    attr = np.mean(A * (W @ basis.D), axis=0)  # (m,)
    if basis.nonneg:
        fire = np.mean(A > 0, axis=0)
        attr = np.where(fire >= min_fire_rate, attr, -np.inf)
    if k is None:  # report how many eligible units have positive attribution
        return int(np.sum(attr > 0))
    order = np.argsort(-attr)
    top = order[:k]
    if not np.all(attr[top] > 0):
        raise RuntimeError(f"{basis.name}: fewer than k={k} units with positive attribution "
                           f"({int(np.sum(attr > 0))} available)")
    return top, attr[top]


# --------------------------------------------------------------------------- persistence
def persistence(A0, At, units, tau, nonneg):
    """Two persistence measures for the selected units on the same fixed inputs.
    frac_active: share of units with mean_t >= tau * mean_t0 (task spec).
    corr: mean Pearson correlation of per-input activations at t vs t0 (protocol definition);
    undefined (constant activations) units are excluded and counted."""
    a0, at = A0[:, units], At[:, units]
    out = {}
    if nonneg:
        m0, mt = a0.mean(0), at.mean(0)
        live = m0 > 0  # a unit selected on train data can still be silent on the test split
        out["n_inactive_t0_test"] = int(np.sum(~live))
        if live.any():
            out["persist_frac_active"] = float(np.mean(mt[live] >= tau * m0[live]))
            out["mean_act_ratio"] = float(np.mean(mt[live] / m0[live]))
        else:
            out["persist_frac_active"] = float("nan")
            out["mean_act_ratio"] = float("nan")
    else:
        out["persist_frac_active"] = float("nan")
        out["mean_act_ratio"] = float("nan")
    s0, st = a0.std(0), at.std(0)
    ok = (s0 > 0) & (st > 0)
    c = np.full(len(units), np.nan)
    c[ok] = np.mean((a0[:, ok] - a0[:, ok].mean(0)) * (at[:, ok] - at[:, ok].mean(0)), axis=0) / (s0[ok] * st[ok])
    out["persist_corr"] = float(np.mean(c[ok])) if ok.any() else float("nan")
    out["n_corr_undefined"] = int(np.sum(~ok))
    return out


def ablation_loads(basis, X, A, y, units, probe, rng):
    """Causal load of the selected units at checkpoint t, using a probe fitted at t.

    Replacement values: 'mean' (unit mean over the evaluation set) and 'resample' (activation
    of a different input, a fixed-point-free permutation). Two spaces:
      recon   : probe applied to the reconstruction x̂ = offset + A D^T vs its ablated version
                (the task specification);
      errpres : the SAE error term is kept, x_abl = x + (A'_S - A_S) D_S^T, baseline = x.
    For the complete neuron and PCA bases the two spaces coincide.
    """
    n = A.shape[0]
    order = rng.permutation(n)
    resample_idx = np.empty(n, dtype=int)
    resample_idx[order] = order[np.roll(np.arange(n), 1)]  # fixed-point free
    assert not np.any(resample_idx == np.arange(n))
    repl = {"mean": np.broadcast_to(A[:, units].mean(0), (n, len(units))),
            "resample": A[resample_idx][:, units]}
    X_hat = basis.reconstruct(A)
    D_S = basis.D[:, units]
    out = {"acc_probe_x": probe_accuracy(probe, X, y), "acc_probe_recon": probe_accuracy(probe, X_hat, y)}
    for rname, R in repl.items():
        delta = (R - A[:, units]) @ D_S.T  # (n, d)
        out[f"acc_abl_{rname}_recon"] = probe_accuracy(probe, X_hat + delta, y)
        out[f"acc_abl_{rname}_errpres"] = probe_accuracy(probe, X + delta, y)
        out[f"load_{rname}_recon"] = out["acc_probe_recon"] - out[f"acc_abl_{rname}_recon"]
        out[f"load_{rname}_errpres"] = out["acc_probe_x"] - out[f"acc_abl_{rname}_errpres"]
    return out


# --------------------------------------------------------------------------- patching
@torch.no_grad()
def patching(model_t0, model_t, split, cfg, device, cil_classes_t, cil_classes_t0, task_cls):
    """Patch each stage's output from the t0 model into the t model on task-1 inputs.
    Also 'head': t0 linear head applied to t penultimate features."""
    model_t0.eval(); model_t.eval()
    logits = {k: [] for k in ("t", "t0", "head", *STAGES)}
    bs = cfg["eval_batch_size"]
    for i in range(0, len(split.labels), bs):
        x = normalize(split.images[i:i + bs].to(device), cfg["norm_mean"], cfg["norm_std"])
        o0 = model_t0.forward_stages(x)
        ot = model_t.forward_stages(x)
        logits["t"].append(ot["logits"].cpu()); logits["t0"].append(o0["logits"].cpu())
        logits["head"].append(model_t0.fc(ot["pen"]).cpu())
        for s in STAGES:
            logits[s].append(model_t.forward_from(s, o0[s]).cpu())
    y = split.labels.numpy()
    L = {k: torch.cat(v).double().numpy() for k, v in logits.items()}
    out = {}
    for k, lg in L.items():
        cil = cil_classes_t0 if k == "t0" else cil_classes_t
        out[f"{k}_til"] = output_accuracy(lg, y, task_cls)
        out[f"{k}_cil"] = output_accuracy(lg, y, cil)
    gap_min = cfg["patch_min_gap"]
    for mode in ("til", "cil"):
        gap = out[f"t0_{mode}"] - out[f"t_{mode}"]
        out[f"gap_{mode}"] = gap
        for k in ("head", *STAGES):
            out[f"recovery_{k}_{mode}"] = ((out[f"{k}_{mode}"] - out[f"t_{mode}"]) / gap
                                           if gap >= gap_min else float("nan"))
    return out
