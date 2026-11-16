"""ResNet-18 for 32x32 inputs (3x3 stem, stride 1, no max-pool), with explicit stages.

Stage outputs are exposed so that activation patching can replace the output of any one
stage. Global average pooling is written as a mean over spatial dims (deterministic on CUDA,
unlike the backward of AdaptiveAvgPool2d).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STAGES = ("stem", "layer1", "layer2", "layer3", "layer4")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CifarResNet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        widths, strides = (64, 128, 256, 512), (1, 2, 2, 2)
        in_planes = 64
        for i, (w, s) in enumerate(zip(widths, strides)):
            layers = [BasicBlock(in_planes, w, s), BasicBlock(w, w, 1)]
            in_planes = w
            setattr(self, f"layer{i + 1}", nn.Sequential(*layers))
        self.fc = nn.Linear(512, n_classes)
        self.feat_dims = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}

    @staticmethod
    def pool(h):
        return h.mean(dim=(2, 3))

    def forward_stages(self, x):
        """Return dict of every stage output, pooled penultimate features and logits."""
        out = {}
        h = x
        for name in STAGES:
            h = getattr(self, name)(h)
            out[name] = h
        out["pen"] = self.pool(h)  # (B, 512) penultimate features
        out["logits"] = self.fc(out["pen"])
        return out

    def forward_from(self, stage, h):
        """Continue the forward pass from the *output* of `stage` (activation patching)."""
        start = STAGES.index(stage) + 1
        for name in STAGES[start:]:
            h = getattr(self, name)(h)
        return self.fc(self.pool(h))

    def forward(self, x):
        return self.forward_stages(x)["logits"]
