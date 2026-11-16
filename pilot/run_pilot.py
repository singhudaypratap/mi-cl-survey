#!/usr/bin/env python
"""Pilot of the vision arm of the MI x CL benchmark protocol (survey Sec. sec:protocol).

One invocation = one (method, seed) run: sequential training on Split CIFAR-10 with a
checkpoint per task, followed by output-, representation-, feature- and circuit-lite-level
measurements at every checkpoint. Resume-safe: finished checkpoints and finished
measurements are detected on disk and skipped.

Example:
    python run_pilot.py --method replay --seed 0 --out runs --device cuda
"""
import argparse
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from micl_pilot import analysis as A
from micl_pilot.data import (load_cifar10, seen_classes, stratified_subset, task_classes,
                             task_split)
from micl_pilot.model import CifarResNet18
from micl_pilot.train import (METHODS, OnlineEWC, ReservoirBuffer, train_one_task,
                              trained_tasks, training_task)

log = logging.getLogger("micl")
HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True, help="root output directory")
    p.add_argument("--device", required=True, help="e.g. cuda, cuda:0, cpu")
    p.add_argument("--config", default=None, help="JSON config (default: configs/full.json, or configs/smoke.json with --smoke)")
    p.add_argument("--epochs-per-task", type=int, default=None, help="override config epochs_per_task")
    p.add_argument("--num-workers", type=int, default=None, help="override config num_workers")
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--smoke", action="store_true", help="~2%% data, 1 epoch, 2 tasks, tiny SAE")
    p.add_argument("--fake-data", action="store_true", help="random tensors with CIFAR shapes (smoke tests only)")
    p.add_argument("--download-only", action="store_true", help="download CIFAR-10 to --data-dir and exit")
    return p.parse_args()


def resolve_config(args):
    path = Path(args.config) if args.config else HERE / "configs" / ("smoke.json" if args.smoke else "full.json")
    cfg = json.loads(path.read_text())
    cfg["_config_file"] = str(path)
    if args.epochs_per_task is not None:
        cfg["epochs_per_task"] = args.epochs_per_task
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    cfg["method"], cfg["seed"] = args.method, args.seed
    cfg["fake_data"], cfg["smoke"] = bool(args.fake_data), bool(args.smoke)
    return cfg


def set_determinism(seed, strict):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def setup_logging(run_dir):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log")):
        h.setFormatter(fmt)
        log.addHandler(h)


def environment(device):
    import sklearn, scipy, pandas, matplotlib, torchvision
    env = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
           "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
           "pandas": pandas.__version__, "matplotlib": matplotlib.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "argv": sys.argv}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    return env


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def load_model(path, device):
    m = CifarResNet18(10).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return m.eval()


# --------------------------------------------------------------------------- analysis
def analyse_stage(stage, model, run_dir, cfg, train, test, device):
    method, seed = cfg["method"], cfg["seed"]
    tasks = trained_tasks(method, stage)
    cil_t = [c for j in tasks for c in task_classes(cfg, j)]
    t0_cls = task_classes(cfg, 0)
    res = {"stage": stage + 1, "trained_tasks": [j + 1 for j in tasks], "output": {}, "probe": {},
           "cka": {}, "capacity": {}, "feature": {}, "patching": {}}

    feats_test, feats_train = {}, {}
    for j in tasks:
        cls = task_classes(cfg, j)
        feats_test[j] = A.extract(model, task_split(test, cls), cfg, device, None)
        feats_train[j] = A.extract(model, task_split(train, cls), cfg, device, None)

    # 1. Output level: CIL (argmax over trained classes) and TIL (argmax over task classes).
    for j in tasks:
        f = feats_test[j]
        res["output"][f"task{j + 1}"] = {"cil": A.output_accuracy(f["logits"], f["y"], cil_t),
                                         "til": A.output_accuracy(f["logits"], f["y"], task_classes(cfg, j))}

    # 2. Representation level: refit vs frozen probes; frozen probe of task j is fitted at the
    #    checkpoint where task j was first learned.
    probe_dir = run_dir / "probes"; probe_dir.mkdir(exist_ok=True)
    refit_pen_task1 = None
    for j in tasks:
        first_stage = j if method != "noshift" else 0
        for layer in ("pen", "mid"):
            ftr, fte = feats_train[j], feats_test[j]
            refit = A.fit_probe(ftr[layer], ftr["y"], cfg)
            ppath = probe_dir / f"frozen_task{j + 1}_{layer}.pt"
            if stage == first_stage:
                atomic_save({"probe": refit}, ppath)
            frozen = torch.load(ppath, weights_only=False)["probe"]
            res["probe"][f"task{j + 1}_{layer}"] = {"refit": A.probe_accuracy(refit, fte[layer], fte["y"]),
                                                     "frozen": A.probe_accuracy(frozen, fte[layer], fte["y"])}
            if j == 0 and layer == "pen":
                refit_pen_task1 = refit

    # t0 reference artifacts
    t0_path = run_dir / "t0_artifacts.pt"
    if stage == 0:
        Xtr0, ytr0 = feats_train[0]["pen"], feats_train[0]["y"]
        g = torch.Generator(device=device).manual_seed(seed * 1000 + 77)
        views = [Xtr0] + [A.extract(model, task_split(train, t0_cls), cfg, device, g)["pen"]
                          for _ in range(cfg["sae_aug_views"])]
        bases, basis_stats = A.build_bases(np.concatenate(views), cfg, device)
        W_fc = model.fc.weight.detach().cpu().double().numpy()
        mfr = cfg.get("min_fire_rate", 0.01)
        avail = {b.name: A.select_units(b, Xtr0, ytr0, W_fc, t0_cls, None, mfr) for b in bases}
        # One k for every basis in this run, so SAE, neuron, PCA and random bases are compared
        # on the same number of units. Capped by the basis with the fewest eligible units.
        k_eff = min(cfg["top_k"], min(avail.values()))
        log.info("eligible units with positive task-1 attribution: %s -> k = %d", avail, k_eff)
        if k_eff < cfg.get("min_k", 8):
            raise RuntimeError(f"too few eligible units for a meaningful selection: {avail}")
        selections = {b.name: A.select_units(b, Xtr0, ytr0, W_fc, t0_cls, k_eff, mfr) for b in bases}
        atomic_save({"bases": bases, "basis_stats": basis_stats, "selections": selections,
                     "test_pen": feats_test[0]["pen"], "test_mid": feats_test[0]["mid"],
                     "test_y": feats_test[0]["y"]}, t0_path)
    t0 = torch.load(t0_path, weights_only=False)
    assert np.array_equal(t0["test_y"], feats_test[0]["y"])
    if stage == 0:
        res["feature_t0_train_stats"] = t0["basis_stats"]

    # Linear CKA between t0 and t features of task 1 test inputs.
    for layer in ("pen", "mid"):
        res["cka"][layer] = A.linear_cka(t0[f"test_{layer}"], feats_test[0][layer])

    # Capacity (plasticity early warning) on the full test set.
    res["capacity"] = A.capacity_stats(A.extract(model, test, cfg, device, None)["pen"])

    # 3. + 4. Feature level with controls, on task-1 test inputs.
    X0, Xt, y1 = t0["test_pen"], feats_test[0]["pen"], feats_test[0]["y"]
    rng = np.random.default_rng([seed, stage, 13])
    for b in t0["bases"]:
        units, attr = t0["selections"][b.name]
        A0, At = b.encode(X0), b.encode(Xt)
        r = {"k": int(len(units)), "attr_t0_mean": float(np.mean(attr))}
        r.update(A.persistence(A0, At, units, cfg["persistence_tau"], b.nonneg))
        st = A.dictionary_stats(b, Xt)
        r.update({"ev_t": st["explained_variance"], "l0_t": st["l0"]})
        r.update(A.ablation_loads(b, Xt, At, y1, units, refit_pen_task1, rng))
        res["feature"][b.name] = r

    # 5. Circuit-lite: stage-wise activation patching t0 -> t on task-1 test inputs.
    model_t0 = load_model(run_dir / "ckpt_stage1.pt", device)
    res["patching"] = A.patching(model_t0, model, task_split(test, t0_cls), cfg, device,
                                 cil_t, task_classes(cfg, 0), t0_cls)
    return res


def acc_bwt(stage_results, mode):
    """ACC and BWT (Lopez-Paz & Ranzato 2017) from R[t][j], over tasks trained by the end."""
    last = stage_results[-1]
    tasks = [int(k[4:]) for k in last["output"]]
    R_final = {j: last["output"][f"task{j}"][mode] for j in tasks}
    first = {}
    for r in stage_results:
        for k, v in r["output"].items():
            first.setdefault(int(k[4:]), v[mode])  # accuracy right after the task was learned
    acc = float(np.mean([R_final[j] for j in tasks]))
    old = [j for j in tasks if j != tasks[-1]] if len(tasks) > 1 else tasks
    bwt = float(np.mean([R_final[j] - first[j] for j in old]))
    return acc, bwt


def flatten(stage_results, cfg):
    rows = []
    base = {"method": cfg["method"], "seed": cfg["seed"]}
    for r in stage_results:
        s = r["stage"]
        for task, v in r["output"].items():
            for mode, x in v.items():
                rows.append({**base, "stage": s, "group": "output", "unit": task, "metric": mode, "value": x})
        for key, v in r["probe"].items():
            for kind, x in v.items():
                rows.append({**base, "stage": s, "group": "probe", "unit": key, "metric": kind, "value": x})
        for layer, x in r["cka"].items():
            rows.append({**base, "stage": s, "group": "cka", "unit": layer, "metric": "cka_t0", "value": x})
        for k, x in r["capacity"].items():
            rows.append({**base, "stage": s, "group": "capacity", "unit": "pen", "metric": k, "value": x})
        for bname, v in r["feature"].items():
            for k, x in v.items():
                rows.append({**base, "stage": s, "group": "feature", "unit": bname, "metric": k, "value": x})
        for k, x in r["patching"].items():
            rows.append({**base, "stage": s, "group": "patching", "unit": "task1", "metric": k, "value": x})
    for mode in ("cil", "til"):
        acc, bwt = acc_bwt(stage_results, mode)
        rows.append({**base, "stage": stage_results[-1]["stage"], "group": "summary", "unit": mode, "metric": "ACC", "value": acc})
        rows.append({**base, "stage": stage_results[-1]["stage"], "group": "summary", "unit": mode, "metric": "BWT", "value": bwt})
    return rows


def main():
    args = parse_args()
    if args.download_only:
        load_cifar10(args.data_dir, fake_data=False, download=True)
        print("CIFAR-10 available in", args.data_dir)
        return
    cfg = resolve_config(args)
    device = torch.device(args.device)
    run_dir = Path(args.out) / f"{cfg['method']}_seed{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    set_determinism(cfg["seed"], cfg["deterministic_strict"])
    cfg_path = run_dir / "config_resolved.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = {k for k in set(old) | set(cfg) if k not in ("_config_file",) and old.get(k) != cfg.get(k)}
        if diff:
            raise RuntimeError(f"existing run in {run_dir} used a different config for keys {sorted(diff)}; "
                               "use a new --out to avoid mixing results")
    atomic_json(cfg, cfg_path)
    atomic_json(environment(device), run_dir / "environment.json")
    log.info("run %s | config %s | device %s | fake_data=%s", run_dir, cfg["_config_file"], device, cfg["fake_data"])
    if cfg["fake_data"]:
        log.warning("FAKE DATA: random tensors, results are for pipeline testing only")

    train_full, test_full = load_cifar10(args.data_dir, cfg["fake_data"], download=False)
    train = stratified_subset(train_full, cfg["data_fraction"], cfg["data_subset_seed"])
    test = stratified_subset(test_full, cfg["data_fraction"], cfg["data_subset_seed"] + 1)
    log.info("data: %d train / %d test images, tasks %s", len(train.labels), len(test.labels),
             [task_classes(cfg, t) for t in range(cfg["n_tasks"])])

    torch.manual_seed(cfg["seed"])
    model = CifarResNet18(10).to(device)
    buffer = ReservoirBuffer(cfg["replay_buffer_size"]) if cfg["method"] == "replay" else None
    ewc = OnlineEWC() if cfg["method"] == "ewc" else None

    stage_results = []
    for stage in range(cfg["n_tasks"]):
        ckpt = run_dir / f"ckpt_stage{stage + 1}.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            if buffer is not None:
                buffer = ReservoirBuffer.from_state(state["buffer"])
            if ewc is not None:
                ewc = OnlineEWC.from_state(state["ewc"])
            log.info("stage %d: checkpoint found, training skipped", stage + 1)
        else:
            t_start = time.time()
            losses = train_one_task(model, train, stage, cfg["method"], cfg, device, cfg["seed"], buffer, ewc)
            atomic_save({"model": model.state_dict(), "losses": losses,
                         "buffer": buffer.state_dict() if buffer is not None else None,
                         "ewc": ewc.state_dict() if ewc is not None else None,
                         "train_task": training_task(cfg["method"], stage) + 1}, ckpt)
            log.info("stage %d trained in %.1f s", stage + 1, time.time() - t_start)
        mpath = run_dir / f"metrics_stage{stage + 1}.json"
        if mpath.exists():
            stage_results.append(json.loads(mpath.read_text()))
            log.info("stage %d: metrics found, analysis skipped", stage + 1)
            continue
        t_start = time.time()
        res = analyse_stage(stage, model.eval(), run_dir, cfg, train, test, device)
        atomic_json(res, mpath)
        stage_results.append(res)
        log.info("stage %d analysed in %.1f s | output %s | patch gap til %.3f",
                 stage + 1, time.time() - t_start, res["output"], res["patching"]["gap_til"])

    import pandas as pd
    rows = flatten(stage_results, cfg)
    pd.DataFrame(rows).to_csv(run_dir / "results_long.csv", index=False)
    summary = {m: dict(zip(("ACC", "BWT"), acc_bwt(stage_results, m))) for m in ("cil", "til")}
    atomic_json({"config": cfg, "summary": summary, "stages": stage_results}, run_dir / "results.json")
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
