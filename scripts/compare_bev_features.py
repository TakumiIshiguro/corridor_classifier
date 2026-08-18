#!/usr/bin/env python3
"""Compares the adopted RGB-only ridge probe (last_cls_regional3, ViT-B/14,
224x224) against variants that add BEV obstacle-scan features (see
scripts/add_bev_to_dataset.py), all via leave-one-session-out CV on the
same features/folds so the numbers are directly comparable.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from corridor_classifier.config import load_training_config, package_root, resolve_path
from corridor_classifier.dataset import CorridorMultiInputDataset, load_dataset_samples
from corridor_classifier.dino_classifier import create_dino_model, resolve_device
from corridor_classifier.linear_probe import (
    bev_zone_features,
    direction_metrics,
    extract_features,
    leave_one_session_out_folds,
)


DEFAULT_L2_CANDIDATES = [0.1, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
MAX_RANGE_M = 8.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--l2-candidates", type=float, nargs="+", default=DEFAULT_L2_CANDIDATES)
    parser.add_argument("--bev-zones", type=int, nargs="+", default=[3, 8, 16])
    return parser.parse_args()


@torch.no_grad()
def collect_rgb_features_and_targets(dino, loader, device, dino_readout):
    feature_batches, target_batches = [], []
    for inputs, targets in loader:
        rgb = inputs["rgb"][:, 0].to(device, non_blocking=True)
        feature_batches.append(extract_features(dino, rgb, dino_readout, None, 0).cpu())
        target_batches.append(targets["directions"])
    return torch.cat(feature_batches), torch.cat(target_batches)


def load_bev_scans(dataset, lateral_bins=64):
    scans = np.full((len(dataset.sequences), lateral_bins, 2), np.nan, dtype=np.float32)
    for index, sequence in enumerate(dataset.sequences):
        sample = sequence[-1]
        if sample.bev_path is None:
            raise ValueError(f"sample has no BEV scan: {sample.image_path}")
        scans[index] = np.load(sample.bev_path)
    return torch.from_numpy(scans)


def run_loso(features, targets, session_names, thresholds, candidates):
    """Leave-one-session-out CV over every l2 candidate, computing one
    eigendecomposition of the training Gram matrix per fold and reusing it
    across all l2 values (O(N^3) once per fold instead of once per
    fold-x-l2), since only the regularization diagonal changes per l2.
    """
    scores_by_l2 = {l2: [] for l2 in candidates}
    for _, train_idx, held_out_idx in leave_one_session_out_folds(session_names):
        train_features = features[train_idx].double()
        train_targets = targets[train_idx].double()
        held_features = features[held_out_idx].double()
        held_targets = targets[held_out_idx]

        feature_mean = train_features.mean(dim=0)
        feature_std = train_features.std(dim=0).clamp_min(1e-6)
        standardized = (train_features - feature_mean) / feature_std
        held_standardized = (held_features - feature_mean) / feature_std

        target_mean = train_targets.mean(dim=0)
        centered_targets = train_targets - target_mean

        gram = standardized @ standardized.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        vt_y = eigenvectors.T @ centered_targets
        kernel_held_v = (held_standardized @ standardized.T) @ eigenvectors

        for l2 in candidates:
            inverse_eigenvalues = 1.0 / (eigenvalues + l2)
            predictions = (
                kernel_held_v * inverse_eigenvalues.unsqueeze(0)
            ) @ vt_y + target_mean
            metrics = direction_metrics(
                predictions.float(), held_targets, thresholds
            )
            scores_by_l2[l2].append(metrics["direction_macro_f1"])

    mean_scores = {l2: sum(v) / len(v) for l2, v in scores_by_l2.items()}
    best_l2 = max(mean_scores, key=mean_scores.get)
    return best_l2, mean_scores[best_l2]


def main():
    args = parse_args()
    config = load_training_config(args.config_dir)
    model_config = config["model"]
    dataset_config = config["dataset"]
    training_config = config["training"]

    device = resolve_device(model_config.get("device", "auto"))
    pretrained_path_config = str(training_config.get("pretrained_weights_path", "")).strip()
    pretrained_path = (
        resolve_path(pretrained_path_config, package_root()) if pretrained_path_config else None
    )
    dino = create_dino_model(
        model_name=model_config["model_name"],
        input_size=model_config["input_size"],
        num_classes=0,
        pretrained=True,
        pretrained_weights_path=pretrained_path,
    ).to(device)
    dino.eval()
    dino_readout = str(model_config.get("dino_readout", "last_cls"))
    thresholds = model_config["direction_thresholds"]

    train_samples = load_dataset_samples(
        resolve_path(dataset_config["train_data_dir"], package_root()),
        model_config["num_classes"],
        dataset_config["train_session_names"],
    )
    test_samples = load_dataset_samples(
        resolve_path(dataset_config["test_data_dir"], package_root()),
        model_config["num_classes"],
        dataset_config["test_session_names"],
    )
    all_samples = list(train_samples) + list(test_samples)

    dataset = CorridorMultiInputDataset(
        all_samples, model_config["input_size"], model_config, sequence_step=1
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
    )
    session_names = [sequence[-1].session_name for sequence in dataset.sequences]
    print(f"sequences={len(dataset)} sessions={sorted(set(session_names))}")

    print("extracting RGB regional3 features...")
    rgb_features, direction_targets = collect_rgb_features_and_targets(
        dino, loader, device, dino_readout
    )
    print("loading BEV scans...")
    bev_scans = load_bev_scans(dataset)
    valid_fraction = (~torch.isnan(bev_scans[..., 0])).float().mean().item()
    print(f"BEV lateral-bin fill rate: {valid_fraction:.2%}")

    results = {}

    l2, score = run_loso(rgb_features, direction_targets, session_names, thresholds, args.l2_candidates)
    results["rgb_only"] = (l2, score, rgb_features.shape[1])

    for zones in args.bev_zones:
        bev_features = bev_zone_features(bev_scans, zones=zones, max_range_m=MAX_RANGE_M)
        combined = torch.cat([rgb_features, bev_features], dim=1)
        l2, score = run_loso(combined, direction_targets, session_names, thresholds, args.l2_candidates)
        results[f"rgb+bev_zones{zones}"] = (l2, score, combined.shape[1])

        l2, score = run_loso(bev_features, direction_targets, session_names, thresholds, args.l2_candidates)
        results[f"bev_only_zones{zones}"] = (l2, score, bev_features.shape[1])

    print("\nLOSO direction macro-F1:")
    print(f"{'condition':>22} {'l2':>10} {'dim':>6} {'macro-F1':>10}")
    for name, (l2, score, dim) in results.items():
        print(f"{name:>22} {l2:>10} {dim:>6} {score:>10.4f}")


if __name__ == "__main__":
    main()
