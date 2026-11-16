"""Sequential training on Split CIFAR-10 with naive fine-tuning, experience replay, online EWC,
and a no-shift control.

All methods share `train_one_task`; they differ only in (i) the extra replay batch and
(ii) the EWC penalty term, which are switched on by `method`.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from .data import augment, make_loader, normalize, task_classes, task_split

log = logging.getLogger("micl")

METHODS = ("naive", "replay", "ewc", "noshift")


def training_task(method, stage):
    """Which task's data is trained at `stage` (0-based).

    The no-shift control keeps training on task 0 for every stage, with the same number of
    epochs, so that drift caused by further training can be separated from drift caused by
    the task switch (protocol, Sec. sec:protocol, "no-shift control").
    """
    return 0 if method == "noshift" else stage


def trained_tasks(method, stage):
    """Tasks whose data the model has been trained on after `stage`."""
    return [0] if method == "noshift" else list(range(stage + 1))


class ReservoirBuffer:
    """Reservoir sampling (Vitter's algorithm R) over raw uint8 stream images."""

    def __init__(self, capacity):
        assert capacity > 0
        self.capacity = capacity
        self.images = torch.zeros((capacity, 3, 32, 32), dtype=torch.uint8)
        self.labels = torch.full((capacity,), -1, dtype=torch.int64)
        self.n_seen = 0

    def __len__(self):
        return min(self.n_seen, self.capacity)

    def add(self, images, labels, rng):
        for i in range(images.shape[0]):
            if self.n_seen < self.capacity:
                j = self.n_seen
            else:
                j = int(rng.integers(0, self.n_seen + 1))
            if j < self.capacity:
                self.images[j] = images[i]
                self.labels[j] = labels[i]
            self.n_seen += 1

    def sample(self, n, generator):
        size = len(self)
        assert size > 0
        idx = torch.randint(0, size, (n,), generator=generator)
        return self.images[idx], self.labels[idx]

    def state_dict(self):
        return {"images": self.images, "labels": self.labels, "n_seen": self.n_seen,
                "capacity": self.capacity}

    @classmethod
    def from_state(cls, s):
        b = cls(s["capacity"])
        b.images, b.labels, b.n_seen = s["images"], s["labels"], s["n_seen"]
        return b


class OnlineEWC:
    """Online EWC (Schwarz et al. 2018) with gamma = 1: Fisher diagonals summed over tasks,
    anchor = parameters at the end of the latest task. Empirical Fisher (true labels),
    per-sample squared gradients, BN in eval mode."""

    def __init__(self):
        self.fisher = None
        self.anchor = None

    def penalty(self, model, lam):
        if self.fisher is None:
            return None
        total = 0.0
        for n, p in model.named_parameters():
            total = total + (self.fisher[n] * (p.float() - self.anchor[n]) ** 2).sum()
        return 0.5 * lam * total

    def consolidate(self, model, split, cfg, device, generator_seed):
        model.eval()
        n = min(cfg["ewc_fisher_samples"], len(split.labels))
        g = torch.Generator().manual_seed(generator_seed)
        idx = torch.randperm(len(split.labels), generator=g)[:n]
        fisher = {k: torch.zeros_like(p, dtype=torch.float32) for k, p in model.named_parameters()}
        for i in idx.tolist():
            x = normalize(split.images[i:i + 1].to(device), cfg["norm_mean"], cfg["norm_std"])
            y = split.labels[i:i + 1].to(device)
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            for k, p in model.named_parameters():
                fisher[k] += p.grad.detach().float() ** 2
        for k in fisher:
            fisher[k] /= n
        if self.fisher is None:
            self.fisher = fisher
        else:
            self.fisher = {k: self.fisher[k] + fisher[k] for k in fisher}
        self.anchor = {k: p.detach().float().clone() for k, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)

    def state_dict(self):
        return {"fisher": self.fisher, "anchor": self.anchor}

    @classmethod
    def from_state(cls, s):
        e = cls()
        e.fisher, e.anchor = s["fisher"], s["anchor"]
        return e


def train_one_task(model, train_split, stage, method, cfg, device, seed, buffer, ewc):
    """Train `model` in place on one stage. Returns per-epoch mean loss."""
    task = training_task(method, stage)
    split = task_split(train_split, task_classes(cfg, task))
    stage_seed = seed * 1000 + stage
    loader = make_loader(split, cfg["batch_size"], True, cfg["num_workers"], stage_seed)
    aug_gen = torch.Generator(device=device).manual_seed(stage_seed)
    replay_gen = torch.Generator().manual_seed(stage_seed + 500)
    reservoir_rng = np.random.default_rng([seed, stage, 7])

    assert cfg["optimizer"] == "sgd" and cfg["lr_schedule"] == "constant_per_task_fresh_optimizer"
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"])
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch_losses = []
    for epoch in range(cfg["epochs_per_task"]):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for images, labels in loader:
            stream_images, stream_labels = images, labels
            # Replay only from stage 2 on, so that the stage-1 checkpoint is identical across
            # naive / replay / EWC for a given seed (shared t0 for all mechanistic metrics).
            if method == "replay" and stage > 0:
                r_img, r_lab = buffer.sample(cfg["replay_batch_size"], replay_gen)
                images = torch.cat([images, r_img])
                labels = torch.cat([labels, r_lab])
            x = normalize(images.to(device, non_blocking=True), cfg["norm_mean"], cfg["norm_std"])
            x = augment(x, cfg["aug_crop_padding"], cfg["aug_hflip"], aug_gen)
            y = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                # Single shared 10-way head, cross-entropy over all outputs (class-incremental).
                loss = F.cross_entropy(logits.float(), y)
            if method == "ewc":
                pen = ewc.penalty(model, cfg["ewc_lambda"])
                if pen is not None:
                    loss = loss + pen
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at stage {stage} epoch {epoch}")
            loss_sum += loss.item()
            n_batches += 1
            if method == "replay":
                buffer.add(stream_images, stream_labels, reservoir_rng)
        epoch_losses.append(loss_sum / n_batches)
        log.info("stage %d (train task %d) epoch %d/%d loss %.4f", stage + 1, task + 1, epoch + 1,
                 cfg["epochs_per_task"], epoch_losses[-1])
    if method == "ewc":
        ewc.consolidate(model, split, cfg, device, stage_seed + 900)
    return epoch_losses
