"""Closed-form ridge-regression linear probe on frozen DINOv2 (+ optional
deterministic depth grid) features.

This is a low-data-first alternative to the SGD-trained GRU/depth-CNN/head
architecture in ``models.py``. Everything upstream of the final linear layer
is either the frozen DINOv2 backbone or a parameter-free deterministic
pooling of the depth map, so the only thing ever "trained" is a ridge
regression closed-form solve. That removes the SGD trajectory (epoch choice,
learning-rate schedule, random init) as a source of run-to-run variance,
which the held-out evaluation in this project showed was comparable in size
to the architecture differences under investigation.
"""
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from corridor_classifier.dino_classifier import build_transform, create_dino_model, resolve_device
from corridor_classifier.models import depth_to_tensor, encode_dino_readout
from corridor_classifier.passage_directions import (
    class_name_from_directions,
    threshold_directions,
)


def bev_zone_features(
    bev_scan: torch.Tensor, zones: int, max_range_m: float
) -> torch.Tensor:
    """Reduces a (B, lateral_bins, 2) BEV nearest-obstacle scan (see
    scripts/add_bev_to_dataset.py; column 0 is forward distance, NaN where
    no obstacle fell in that lateral bin) to a (B, zones) feature: the mean
    forward clearance within each left-to-right zone, with empty zones
    treated as fully open (max_range_m). Zones split the scan by lateral
    bin index, so with the adopted 3 zones this is directly analogous to
    the front/left/right split the model predicts.
    """
    if bev_scan.ndim != 3 or bev_scan.shape[2] != 2:
        raise ValueError("bev_scan must have shape (batch, lateral_bins, 2)")
    forward = bev_scan[..., 0]
    batch_size, bins = forward.shape
    features = []
    for band in forward.tensor_split(int(zones), dim=1):
        valid = ~torch.isnan(band)
        count = valid.sum(dim=1).clamp_min(1)
        summed = torch.nan_to_num(band, nan=0.0).sum(dim=1)
        mean_forward = summed / count
        all_empty = ~valid.any(dim=1)
        mean_forward = torch.where(
            all_empty, torch.full_like(mean_forward, max_range_m), mean_forward
        )
        features.append(mean_forward)
    return torch.stack(features, dim=1)


def depth_grid_features(depth: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Deterministically pools a (B, 2, H, W) depth tensor (normalized
    log-depth + validity mask, see ``models.depth_to_tensor``) to a
    (B, 2 * grid_size * grid_size) feature vector. No learned parameters.
    """
    if depth.ndim != 4 or depth.shape[1] != 2:
        raise ValueError("depth must have shape (batch, 2, height, width)")
    pooled = F.adaptive_avg_pool2d(depth, (int(grid_size), int(grid_size)))
    return pooled.flatten(1)


@torch.no_grad()
def extract_features(
    dino: nn.Module,
    rgb: torch.Tensor,
    dino_readout: str,
    depth: Optional[torch.Tensor] = None,
    depth_grid_size: int = 4,
) -> torch.Tensor:
    """Extracts one flat feature vector per sample from a single RGB frame
    (and optional depth frame). No gradients, no trainable parameters.
    """
    rgb_features = encode_dino_readout(dino, rgb, dino_readout)
    if depth is None:
        return rgb_features
    return torch.cat(
        [rgb_features, depth_grid_features(depth, depth_grid_size)],
        dim=-1,
    )


class RidgeProbe:
    """Closed-form (dual/kernel) ridge regression classifier.

    Targets are treated as regression values in {0, 1} per output (the
    standard "ridge classifier" recipe). Solved in the dual form
    ``(X X^T + l2 * I) alpha = Y`` because the feature dimension (DINOv2
    features concatenated across regions) is typically larger than the
    number of training sequences, so the dual system is the smaller of the
    two to invert.
    """

    def __init__(self, l2: float):
        self.l2 = float(l2)
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        self.target_mean: Optional[torch.Tensor] = None
        self.train_features: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None

    def fit(self, features: torch.Tensor, targets: torch.Tensor) -> "RidgeProbe":
        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError("features and targets must be 2D")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have matching batch size")
        features = features.double()
        targets = targets.double()
        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0).clamp_min(1e-6)
        standardized = (features - self.feature_mean) / self.feature_std
        self.target_mean = targets.mean(dim=0)
        centered_targets = targets - self.target_mean

        num_samples = standardized.shape[0]
        gram = standardized @ standardized.T
        gram = gram + self.l2 * torch.eye(num_samples, dtype=torch.float64)
        self.alpha = torch.linalg.solve(gram, centered_targets)
        self.train_features = standardized
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            raise RuntimeError("RidgeProbe must be fit before predict")
        standardized = (
            features.double() - self.feature_mean
        ) / self.feature_std
        kernel = standardized @ self.train_features.T
        return (kernel @ self.alpha + self.target_mean).float()


def binary_f1(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    true_positive = int((predictions & targets).sum())
    false_positive = int((predictions & ~targets).sum())
    false_negative = int((~predictions & targets).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def direction_metrics(
    predicted_probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Sequence[float],
) -> Dict[str, float]:
    thresholds_tensor = torch.as_tensor(thresholds, dtype=predicted_probabilities.dtype)
    predictions = predicted_probabilities >= thresholds_tensor
    targets_bool = targets.bool()
    direction_f1 = [
        binary_f1(predictions[:, index], targets_bool[:, index])
        for index in range(predictions.shape[1])
    ]
    return {
        "front_f1": direction_f1[0],
        "left_f1": direction_f1[1],
        "right_f1": direction_f1[2],
        "direction_macro_f1": sum(direction_f1) / len(direction_f1),
        "direction_exact_accuracy": float(
            (predictions == targets_bool).all(dim=1).float().mean()
        ),
    }


def select_l2_by_validation(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    thresholds: Sequence[float],
    candidates: Sequence[float],
) -> Tuple[float, Dict[str, float], RidgeProbe]:
    """Fits a ridge probe for each candidate l2 and returns the one with the
    best validation direction macro-F1, along with its metrics and probe.
    """
    best = None
    for l2 in candidates:
        probe = RidgeProbe(l2).fit(train_features, train_targets)
        predictions = probe.predict(val_features)
        metrics = direction_metrics(predictions, val_targets, thresholds)
        if best is None or metrics["direction_macro_f1"] > best[1]["direction_macro_f1"]:
            best = (l2, metrics, probe)
    return best


def leave_one_session_out_folds(
    session_names: Sequence[str],
) -> Iterator[Tuple[str, List[int], List[int]]]:
    unique_sessions = sorted(set(session_names))
    for held_out in unique_sessions:
        train_indices = [i for i, name in enumerate(session_names) if name != held_out]
        held_out_indices = [i for i, name in enumerate(session_names) if name == held_out]
        yield held_out, train_indices, held_out_indices


def select_l2_by_leave_one_session_out(
    features: torch.Tensor,
    targets: torch.Tensor,
    session_names: Sequence[str],
    thresholds: Sequence[float],
    candidates: Sequence[float],
) -> Tuple[float, Dict[float, float]]:
    """Picks the l2 that maximizes the mean direction macro-F1 across
    leave-one-session-out folds. Reuses one feature extraction pass across
    every fold and every candidate, since only the closed-form solve
    changes per fold/candidate.
    """
    scores_by_l2: Dict[float, float] = {}
    for l2 in candidates:
        fold_scores = []
        for _, train_indices, held_out_indices in leave_one_session_out_folds(session_names):
            probe = RidgeProbe(l2).fit(features[train_indices], targets[train_indices])
            predictions = probe.predict(features[held_out_indices])
            metrics = direction_metrics(
                predictions, targets[held_out_indices], thresholds
            )
            fold_scores.append(metrics["direction_macro_f1"])
        scores_by_l2[l2] = sum(fold_scores) / len(fold_scores)
    best_l2 = max(scores_by_l2, key=scores_by_l2.get)
    return best_l2, scores_by_l2


@dataclass(frozen=True)
class PassageDirectionPrediction:
    direction_scores: tuple
    open_directions: tuple
    class_name: str


class RidgeLinearProbePredictor:
    """Loads a checkpoint saved by scripts/fit_production_linear_probe.py
    (or train_linear_probe.py) and runs single-frame inference: frozen
    DINOv2 forward pass + deterministic depth pooling (if used) + the
    closed-form ridge probe. Stateless across frames, no temporal buffer.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.device = resolve_device(device)
        self.dino_readout = str(checkpoint["dino_readout"])
        self.use_depth = bool(checkpoint["use_depth"])
        self.depth_grid_size = int(checkpoint["depth_grid_size"])
        self.depth_min_m = float(checkpoint.get("depth_min_m", 0.1))
        self.depth_max_m = float(checkpoint.get("depth_max_m", 10.0))
        self.direction_thresholds = tuple(
            float(value) for value in checkpoint["direction_thresholds"]
        )
        self.class_names = tuple(checkpoint["class_names"])
        self.input_size = tuple(int(value) for value in checkpoint["input_size"])
        self.transform = build_transform(list(self.input_size))

        self.probe = RidgeProbe(float(checkpoint["l2"]))
        self.probe.feature_mean = checkpoint["feature_mean"].to(self.device)
        self.probe.feature_std = checkpoint["feature_std"].to(self.device)
        self.probe.target_mean = checkpoint["target_mean"].to(self.device)
        self.probe.train_features = checkpoint["train_features"].to(self.device)
        self.probe.alpha = checkpoint["alpha"].to(self.device)

        pretrained_weights_path = checkpoint.get("pretrained_weights_path")
        if not pretrained_weights_path:
            raise ValueError(
                "checkpoint is missing pretrained_weights_path: the frozen "
                "DINOv2 backbone must be reloaded with the exact weights "
                "used during feature extraction, or the probe's features "
                "will not match"
            )
        self.dino = create_dino_model(
            model_name=str(checkpoint["model_name"]),
            input_size=list(self.input_size),
            num_classes=0,
            pretrained=True,
            pretrained_weights_path=str(pretrained_weights_path),
        ).to(self.device)
        self.dino.eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        depth_meters=None,
    ) -> PassageDirectionPrediction:
        rgb_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        depth_tensor = None
        if self.use_depth:
            if depth_meters is None:
                raise ValueError("depth input is required by this checkpoint")
            depth_tensor = depth_to_tensor(
                depth_meters, self.depth_min_m, self.depth_max_m
            ).unsqueeze(0).to(self.device)
        features = extract_features(
            self.dino, rgb_tensor, self.dino_readout, depth_tensor, self.depth_grid_size
        )
        scores = self.probe.predict(features)[0]
        direction_scores = tuple(float(value) for value in scores.cpu().tolist())
        open_directions = threshold_directions(direction_scores, self.direction_thresholds)
        return PassageDirectionPrediction(
            direction_scores=direction_scores,
            open_directions=open_directions,
            class_name=class_name_from_directions(open_directions),
        )
