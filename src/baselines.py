"""Compact baseline models used for manuscript-level comparison.

These models are intentionally kept in one file to avoid a confusing repository
structure while still supporting D-CNN, D-RNN, D-DNN, and D-ConvNet baselines.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class DCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class DDNN(nn.Module):
    def __init__(self, in_channels: int, time_points: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * time_points, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class DRNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 2, hidden: int = 96):
        super().__init__()
        self.rnn = nn.GRU(input_size=in_channels, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class DConvNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_baseline(name: str, in_channels: int, time_points: int, n_classes: int = 2) -> nn.Module:
    key = name.lower().replace("-", "").replace("_", "")
    if key == "dcnn":
        return DCNN(in_channels, n_classes)
    if key == "ddnn":
        return DDNN(in_channels, time_points, n_classes)
    if key == "drnn":
        return DRNN(in_channels, n_classes)
    if key in {"dconvnet", "convnet"}:
        return DConvNet(in_channels, n_classes)
    raise ValueError(f"Unknown baseline: {name}")
