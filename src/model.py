"""DGANM model definitions.

The implementation follows the manuscript description: a classifier-guided
adversarial domain adaptation model with generator, discriminator, and classifier
modules using one-dimensional convolutional EEG representations.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, activation: str = "relu"):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Generator(nn.Module):
    """Source-to-target EEG representation generator.

    The output preserves the input shape [batch, channels, time], allowing direct
    classifier supervision and discriminator-based domain alignment.
    """

    def __init__(self, in_channels: int, filters: Sequence[int] = (64, 128, 256), kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_channels
        for width in filters:
            layers.append(ConvBNAct(last, int(width), kernel_size=kernel_size, activation="relu"))
            last = int(width)
        layers.append(nn.Dropout(p=float(dropout)))
        layers.append(nn.Conv1d(last, in_channels, kernel_size=1, stride=1, padding=0))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    """Domain discriminator for real target vs generated target-like EEG."""

    def __init__(self, in_channels: int, filters: Sequence[int] = (64, 128, 256), kernel_size: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_channels
        for width in filters:
            layers.append(ConvBNAct(last, int(width), kernel_size=kernel_size, activation="leaky_relu"))
            last = int(width)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(last, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.pool(z)
        return self.head(z).view(-1)


class Classifier(nn.Module):
    """Seizure/non-seizure classifier used both for prediction and generator guidance."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int = 2,
        filters: Sequence[int] = (64, 128),
        hidden: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_channels
        for width in filters:
            layers.append(ConvBNAct(last, int(width), kernel_size=kernel_size, activation="relu"))
            last = int(width)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(last, int(hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout) / 2.0),
            nn.Linear(int(hidden), int(n_classes)),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        z = self.features(x)
        pooled = self.pool(z).flatten(1)
        logits = self.head(pooled)
        if return_features:
            return logits, pooled
        return logits


class DGANM(nn.Module):
    """Container module holding all DGANM components."""

    def __init__(self, cfg: dict):
        super().__init__()
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        in_channels = int(data_cfg.get("n_channels", 19))
        n_classes = int(data_cfg.get("n_classes", 2))
        kernel_size = int(model_cfg.get("kernel_size", 3))
        dropout = float(model_cfg.get("dropout", 0.5))
        self.generator = Generator(
            in_channels=in_channels,
            filters=model_cfg.get("generator_filters", [64, 128, 256]),
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            in_channels=in_channels,
            filters=model_cfg.get("discriminator_filters", [64, 128, 256]),
            kernel_size=kernel_size,
        )
        self.classifier = Classifier(
            in_channels=in_channels,
            n_classes=n_classes,
            filters=model_cfg.get("classifier_filters", [64, 128]),
            hidden=int(model_cfg.get("classifier_hidden", 128)),
            kernel_size=kernel_size,
            dropout=dropout,
        )


def gaussian_mmd(x: torch.Tensor, y: torch.Tensor, sigmas: Iterable[float] = (1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """Multi-kernel maximum mean discrepancy used as a soft feature alignment term."""
    if x.ndim > 2:
        x = x.flatten(1)
    if y.ndim > 2:
        y = y.flatten(1)
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    loss = torch.zeros((), device=x.device)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * float(sigma) ** 2)
        loss = loss + torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean() - 2.0 * torch.exp(-gamma * xy).mean()
    return loss / max(1, len(tuple(sigmas)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
