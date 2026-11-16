"""Split CIFAR-10 data handling.

All images are held in memory as uint8 tensors of shape (N, 3, 32, 32). Augmentation and
normalisation are applied on the compute device by `augment` / `normalize`, which are shared
by the stream and the replay buffer so that every method sees identical preprocessing.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

CIFAR_TRAIN_N = 50000
CIFAR_TEST_N = 10000
N_CLASSES = 10


@dataclass(frozen=True)
class Split:
    images: torch.Tensor  # uint8, (N, 3, 32, 32)
    labels: torch.Tensor  # int64, (N,), original CIFAR-10 class ids


def _from_torchvision(data_dir, train, download):
    import torchvision
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=download)
    images = torch.from_numpy(np.ascontiguousarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(ds.targets, dtype=torch.int64)
    assert images.shape[1:] == (3, 32, 32) and images.dtype == torch.uint8
    return Split(images, labels)


def _fake(n, seed):
    """Random tensors with CIFAR-10 shapes, for pipeline smoke tests only.

    A weak class-dependent colour offset is added so that classifiers and probes have
    something non-trivial to fit; the numbers it produces carry no scientific meaning.
    """
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n, dtype=torch.int64) % N_CLASSES
    labels = labels[torch.randperm(n, generator=g)]
    noise = torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.int16)
    class_offset = torch.randint(-60, 61, (N_CLASSES, 3, 1, 1), dtype=torch.int16,
                                 generator=torch.Generator().manual_seed(1234))
    images = (noise + class_offset[labels]).clamp(0, 255).to(torch.uint8)
    return Split(images, labels)


def load_cifar10(data_dir, fake_data, download):
    if fake_data:
        return _fake(CIFAR_TRAIN_N, seed=0), _fake(CIFAR_TEST_N, seed=1)
    return _from_torchvision(data_dir, True, download), _from_torchvision(data_dir, False, download)


def stratified_subset(split, fraction, seed):
    """Keep `fraction` of each class, chosen with a fixed seed shared by all runs."""
    assert 0.0 < fraction <= 1.0, fraction
    if fraction == 1.0:
        return split
    rng = np.random.default_rng(seed)
    keep = []
    labels = split.labels.numpy()
    for c in range(N_CLASSES):
        idx = np.flatnonzero(labels == c)
        n_keep = int(round(fraction * len(idx)))
        assert n_keep >= 2, f"fraction {fraction} leaves {n_keep} samples of class {c}"
        keep.append(np.sort(rng.choice(idx, size=n_keep, replace=False)))
    keep = torch.from_numpy(np.sort(np.concatenate(keep)))
    return Split(split.images[keep], split.labels[keep])


def task_classes(cfg, task):
    """Class ids of task `task` (0-based)."""
    k = cfg["classes_per_task"]
    order = cfg["class_order"]
    assert sorted(order) == list(range(N_CLASSES)), order
    assert cfg["n_tasks"] * k <= N_CLASSES
    return order[task * k:(task + 1) * k]


def seen_classes(cfg, task):
    """All classes of tasks 0..task (inclusive)."""
    return [c for t in range(task + 1) for c in task_classes(cfg, t)]


def task_split(split, classes):
    mask = torch.isin(split.labels, torch.tensor(classes, dtype=torch.int64))
    assert mask.any(), classes
    return Split(split.images[mask], split.labels[mask])


def normalize(images_uint8, mean, std):
    """uint8 (B,3,32,32) on any device -> float32 normalised."""
    x = images_uint8.float().div_(255.0)
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def augment(x, padding, hflip, generator):
    """Random crop with zero padding + horizontal flip, on float tensors (B,3,H,W).

    `generator` must live on the same device as x; it is the only source of randomness.
    """
    b, _, h, w = x.shape
    if padding > 0:
        xp = F.pad(x, (padding, padding, padding, padding))
        dx = torch.randint(0, 2 * padding + 1, (b,), device=x.device, generator=generator)
        dy = torch.randint(0, 2 * padding + 1, (b,), device=x.device, generator=generator)
        rows = (dy.view(b, 1) + torch.arange(h, device=x.device).view(1, h))  # (b, h)
        cols = (dx.view(b, 1) + torch.arange(w, device=x.device).view(1, w))  # (b, w)
        bidx = torch.arange(b, device=x.device).view(b, 1, 1)
        xp = xp.permute(0, 2, 3, 1)  # (b, H, W, c)
        x = xp[bidx, rows.view(b, h, 1), cols.view(b, 1, w)].permute(0, 3, 1, 2).contiguous()
    if hflip:
        flip = torch.rand(b, device=x.device, generator=generator) < 0.5
        x = torch.where(flip.view(b, 1, 1, 1), x.flip(3), x)
    return x


class IndexDataset(torch.utils.data.Dataset):
    """Yields (uint8 image, label). Randomness lives on the main process, not in workers."""

    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split.labels)

    def __getitem__(self, i):
        return self.split.images[i], self.split.labels[i]


def make_loader(split, batch_size, shuffle, num_workers, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        IndexDataset(split), batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, generator=g, pin_memory=False,
        persistent_workers=False)
