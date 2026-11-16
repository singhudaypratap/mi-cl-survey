# MI x CL pilot: vision arm of the benchmark protocol (reduced form)

Reduced pilot of the vision testbed in Section "The evaluation gap" (`sec:protocol`) of the survey.
Split CIFAR-10 (5 tasks x 2 classes, class-incremental, single shared 10-way head),
CIFAR ResNet-18 (3x3 stem, no max-pool), conditions `naive`, `replay` (reservoir, 500 samples),
`ewc` (online EWC) and the protocol's `noshift` control, seeds 0, 1, 2.

## What is measured at every checkpoint t (task 1 = checkpoint t0)

| Level | Quantity | Where |
|---|---|---|
| Output | accuracy matrix (class-incremental, argmax over trained classes, and task-aware 2-way), ACC, BWT | `output`, `summary` |
| Representation | linear probe re-fitted at t (decodable) vs probe fitted when the task was learned and frozen (read out), penultimate (512-d) and layer3 (256-d, pooled) features, every trained task; linear CKA(t0, t) on task-1 test inputs; dead-unit fraction and effective rank | `probe`, `cka`, `capacity` |
| Feature | ReLU SAE (8x, L1) fitted on t0 penultimate features of task-1 training images, then frozen. Top-k (k=32) units by attribution to the task-1 correct-vs-other logit difference at t0 (selected on train, evaluated on test). Explained variance at t; persistence as (a) fraction of units with mean activation >= tau x t0 value (tau=0.5) and (b) mean Pearson correlation with t0 activations on the same inputs (protocol definition); causal load = drop in accuracy of a probe fitted at t when the k units are mean- or resample-ablated, both in the reconstruction (`recon`) and with the SAE error term kept (`errpres`) | `feature` |
| Controls | neuron basis (top-k raw neurons), random dictionary of the same size (identical construction, untrained), PCA basis (complete, 512 components), 3 SAE seeds (+3 random-dictionary seeds), no-shift condition, frozen-vs-refit probes | `feature/<basis>` |
| Circuit-lite | output of each stage (stem, layer1-4) from the t0 model patched into the model at t on task-1 test inputs; also the t0 head on t features. Recovery = (acc_patched - acc_t) / (acc_t0 - acc_t), NaN when the gap is < 0.02 | `patching` |

## Files

```
run_pilot.py          single entry point, one (method, seed) run, resume-safe
run_all.sh            all runs on one GPU, PARALLEL at a time, then aggregation
aggregate.py          tables (CSV + LaTeX, mean ± sd over seeds) and 3 figures
configs/full.json     every parameter of the full run (recorded per run in config_resolved.json)
configs/smoke.json    ~2% data, 2 tasks, 1 epoch, tiny SAE
micl_pilot/           data.py, model.py, train.py, analysis.py
tests/test_core.py    identity checks (CKA, crop/flip, reservoir uniformity, ablation, patching)
```

Per run, `<out>/<method>_seed<k>/` holds `ckpt_stage{t}.pt`, `metrics_stage{t}.json`,
`results.json`, `results_long.csv`, `run.log`, `config_resolved.json`, `environment.json`,
`t0_artifacts.pt` (frozen dictionaries, unit selections, t0 features) and `probes/`.
`runs/aggregate/` gets `aggregate_long.csv`, `table_summary.{csv,tex}`,
`fig_persistence.{pdf,png}`, `fig_probes.{pdf,png}`, `fig_patching.{pdf,png}`.

## Server commands (run from this directory)

All commands are backgrounded; check each log before starting the next step.

```bash
# 1. virtual environment
nohup python3 -m venv .venv > venv.log 2>&1 & disown

# 2. dependencies (if the pinned torch does not match your driver, first run e.g.
#    nohup .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 > torch.log 2>&1 & disown
#    and then remove the torch/torchvision lines from requirements.txt)
nohup .venv/bin/pip install -r requirements.txt > pip.log 2>&1 & disown

# 3. unit checks and GPU smoke test on real CIFAR-10 (downloads to ./data, ~2 min)
nohup .venv/bin/python tests/test_core.py > tests.log 2>&1 & disown
nohup env PY=.venv/bin/python OUT=runs_smoke DEVICE=cuda:0 EPOCHS=1 SEEDS="0 1" PARALLEL=3 EXTRA="--smoke" bash run_all.sh > smoke.log 2>&1 & disown

# 4. full run: 4 conditions x 3 seeds, 20 epochs/task, 3 runs at a time on one GPU
nohup env PY=.venv/bin/python OUT=runs DEVICE=cuda:0 EPOCHS=20 PARALLEL=3 bash run_all.sh > run_all.log 2>&1 & disown

# 5. monitoring
nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 > gpu.log 2>&1 & disown
tail -f run_all.log runs/logs/*.log gpu.log

# 6. aggregation (run_all.sh already does this at the end; rerun after a partial or resumed run)
nohup .venv/bin/python aggregate.py --runs runs --out runs/aggregate > aggregate.log 2>&1 & disown
```

A single run, e.g. for debugging:

```bash
nohup .venv/bin/python run_pilot.py --method replay --seed 0 --epochs-per-task 20 --out runs --device cuda:0 --num-workers 4 > replay0.log 2>&1 & disown
```

Only the requested 9 runs (no no-shift control): add `METHODS="naive replay ewc"` to the `env` list.
Stop everything: `pkill -f run_pilot.py`.

## Resume

Re-launch the same command. Finished checkpoints (`ckpt_stage{t}.pt`) and finished measurements
(`metrics_stage{t}.json`) are skipped; training of each stage is reseeded from (seed, stage), so a
resumed run reproduces an uninterrupted one. A run directory refuses to resume under a different
config (e.g. another `EPOCHS`); use a new `OUT` instead.

## Reproducibility

Seeds are fixed per run and per stage; cuDNN is deterministic and `torch.use_deterministic_algorithms(True)`
is on (`deterministic_strict` in the config; if an op raises a determinism error on your GPU,
set it to `false` and record that). Results are bit-identical only for the same thread count
(`OMP_NUM_THREADS` is set by `run_all.sh`) and hardware. AMP (fp16) is used for training on CUDA only;
all measurements are computed in fp32/fp64.

## Estimated runtime (RTX 8000, 20 epochs/task)

Not measured on the GPU; estimate from throughput of CIFAR ResNet-18 with AMP (~3-4k img/s
under deterministic cuDNN): naive/noshift ~5-7 min of training, EWC ~7-9 min, replay ~10-14 min
(double batch), plus ~3-5 min of measurements per run (SAE fitting at t0, ~30 probe fits).
12 runs: ~2-3 h sequentially (`PARALLEL=1`), ~1-1.5 h with `PARALLEL=3`
(each run needs < 3 GB of GPU memory). The 9 requested runs alone take ~3/4 of that.
Check `stage 1 trained in ... s` in the first log and scale.

## Before reporting numbers: values to check

* `sae_l1` (0.004) is **untuned**. At t0, `metrics_stage1.json -> feature_t0_train_stats.sae_s*`
  should show explained variance > ~0.8 and L0 of roughly 10-60 active latents; adjust and rerun
  if not (changing it changes every SAE result).
* `ewc_lambda` (100) is **untuned**; online EWC is known to be weak in class-incremental settings.
* lr 0.03, batch 128, constant lr with a fresh SGD optimizer per task, cross-entropy over all 10
  outputs: standard choices, not tuned.
* Smoke-test outputs (`--smoke`, and especially `--fake-data`) carry no scientific meaning;
  the table caption is marked when fake data was used.

## Deviations from the protocol text (sec:protocol)

* Reduced testbed: Split CIFAR-10 (5 tasks) and ResNet-18 only; no CIFAR-100, ViT or LwF.
* Recoverability (c), the fixed relearning budget, is not implemented; (a) is the re-fitted probe
  (head-only readout), (b) is stage-wise patching (stage outputs, not individual units).
* The random-model SAE control of the protocol (an SAE trained on a randomly initialised network) is
  replaced by an untrained random dictionary of the same size, as requested for the pilot.
* Units are selected at t0 only; dictionaries fitted at the final checkpoint and applied
  backwards are not included.
* Feature-level metrics are computed on the penultimate layer for task 1 only.
* Replay starts at task 2 (the buffer fills during task 1), so naive, replay and EWC share the
  identical t0 checkpoint for a given seed.
