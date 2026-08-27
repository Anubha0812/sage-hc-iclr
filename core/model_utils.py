"""Model construction, losses, reproducibility, and evaluation for SAGE-HC."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from networks.rggcn import ResidualGatedGCN


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(
    model_name: str,
    num_features: int,
    device: torch.device,
    model_kwargs: dict[str, Any],
) -> nn.Module:
    if model_name != "rggcn":
        raise ValueError(
            f"This release contains the paper model only; expected model_name='rggcn', got {model_name!r}."
        )
    return ResidualGatedGCN(num_features=num_features, **model_kwargs).to(device)


def make_criterion(name: str, beta: float = 0.1) -> nn.Module:
    name = name.lower()
    if name == "smooth_l1":
        return nn.SmoothL1Loss(beta=beta)
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported criterion: {name!r}")


def accuracy(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.1) -> float:
    # Strict inequality preserves the evaluation behavior used in the reported runs.
    return float((np.abs(preds - labels) < threshold).sum() / len(labels))


def evaluate_model(model, loader, device, criterion, loss2):
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index).clamp(0.0, 1.0)
            labels = batch.y.unsqueeze(1)
            total_loss += criterion(predictions, labels).item()
            total_l1 += loss2(predictions, labels).item()
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())

    if not all_preds:
        raise ValueError("Cannot evaluate an empty loader.")

    preds = torch.cat(all_preds).squeeze().numpy()
    labels = torch.cat(all_labels).squeeze().numpy()
    return (
        total_loss / len(loader),
        total_l1 / len(loader),
        accuracy(preds, labels, 0.1),
        accuracy(preds, labels, 0.2),
        preds,
        labels,
    )
