#!/usr/bin/env python3
"""Fits a closed-form ridge-regression linear probe on frozen DINOv2 (+
optional deterministic depth grid) features for the passage_directions task.

This is a low-data-first alternative to the SGD-trained GRU/depth-CNN
architecture in train.py: everything upstream of the final linear layer is
frozen or parameter-free, so there is no training trajectory (epoch choice,
learning-rate schedule, random init) to overfit the checkpoint-selection
set with. Regularization strength is chosen by evaluating on a validation
session split, matching the honest val/held-out protocol established in
evaluate_session_holdout.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch
from torch.utils.data import DataLoader

from corridor_classifier.config import load_training_config, package_root, resolve_path
from corridor_classifier.dataset import CorridorMultiInputDataset, load_dataset_samples
from corridor_classifier.dino_classifier import create_dino_model, resolve_device
from corridor_classifier.linear_probe import direction_metrics, extract_features, select_l2_by_validation


DEFAULT_L2_CANDIDATES = [0.1, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument(
        "--val-session-names",
        required=True,
        nargs="+",
        help="Sessions used only to pick the ridge regularization strength.",
    )
    parser.add_argument(
        "--heldout-session-names",
        nargs="+",
        default=[],
        help="Sessions never used for fitting or regularization selection; "
        "evaluated once at the end if given.",
    )
    parser.add_argument("--depth-grid-size", type=int, default=4)
    parser.add_argument("--l2-candidates", type=float, nargs="+", default=DEFAULT_L2_CANDIDATES)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--dino-readout-override",
        default="",
        help="Overrides model.dino_readout for quick ablations without a new config dir.",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Ignores model.use_depth and skips depth features entirely.",
    )
    return parser.parse_args()


def build_loader(samples, model_config, dataset_config, batch_size):
    dataset = CorridorMultiInputDataset(
        samples,
        model_config["input_size"],
        model_config,
        sequence_step=dataset_config.get("test_sequence_step", 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
    )
    return dataset, loader


@torch.no_grad()
def collect_features(dino, loader, device, dino_readout, use_depth, depth_grid_size):
    feature_batches = []
    target_batches = []
    for inputs, targets in loader:
        rgb = inputs["rgb"][:, 0].to(device, non_blocking=True)
        depth = inputs["depth"][:, 0].to(device, non_blocking=True) if use_depth else None
        features = extract_features(dino, rgb, dino_readout, depth, depth_grid_size)
        feature_batches.append(features.cpu())
        target_batches.append(targets["directions"])
    return torch.cat(feature_batches), torch.cat(target_batches)


def load_session_features(
    session_names, dataset_config, model_config, device, dino, dino_readout, use_depth, depth_grid_size, batch_size
):
    samples = load_dataset_samples(
        resolve_path(dataset_config["test_data_dir"], package_root()),
        model_config["num_classes"],
        session_names,
    )
    _, loader = build_loader(samples, model_config, dataset_config, batch_size)
    features, targets = collect_features(dino, loader, device, dino_readout, use_depth, depth_grid_size)
    return samples, features, targets


def main():
    args = parse_args()
    config = load_training_config(args.config_dir)
    model_config = config["model"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    if model_config.get("output_mode") != "passage_directions":
        raise ValueError("linear probe training requires passage_directions output mode")
    if int(model_config.get("sequence_length", 1)) != 1:
        raise ValueError("linear probe training requires model.sequence_length == 1")
    if args.dino_readout_override:
        model_config["dino_readout"] = args.dino_readout_override

    device = resolve_device(model_config.get("device", "auto"))
    pretrained_path = resolve_path(training_config["pretrained_weights_path"], package_root())
    dino = create_dino_model(
        model_name=model_config["model_name"],
        input_size=model_config["input_size"],
        num_classes=0,
        pretrained=True,
        pretrained_weights_path=pretrained_path,
    ).to(device)
    dino.eval()

    dino_readout = str(model_config.get("dino_readout", "last_cls"))
    use_depth = bool(model_config["use_depth"]) and not args.no_depth
    model_config["use_depth"] = use_depth
    batch_size = int(training_config["batch_size"])

    train_samples = load_dataset_samples(
        resolve_path(dataset_config["train_data_dir"], package_root()),
        model_config["num_classes"],
        dataset_config["train_session_names"],
    )
    _, train_loader = build_loader(train_samples, model_config, dataset_config, batch_size)
    print(f"extracting train features: {len(train_samples)} samples")
    train_features, train_targets = collect_features(
        dino, train_loader, device, dino_readout, use_depth, args.depth_grid_size
    )

    print(f"extracting val features: sessions={args.val_session_names}")
    val_samples, val_features, val_targets = load_session_features(
        args.val_session_names, dataset_config, model_config, device, dino,
        dino_readout, use_depth, args.depth_grid_size, batch_size,
    )
    print(
        f"feature_dim={train_features.shape[1]} train_n={train_features.shape[0]} "
        f"val_n={val_features.shape[0]}"
    )

    best_l2, val_metrics, probe = select_l2_by_validation(
        train_features, train_targets, val_features, val_targets,
        model_config["direction_thresholds"], args.l2_candidates,
    )
    print(f"best_l2={best_l2}")
    print(f"val_metrics={val_metrics}")

    if args.heldout_session_names:
        print(f"extracting heldout features: sessions={args.heldout_session_names}")
        heldout_samples, heldout_features, heldout_targets = load_session_features(
            args.heldout_session_names, dataset_config, model_config, device, dino,
            dino_readout, use_depth, args.depth_grid_size, batch_size,
        )
        heldout_predictions = probe.predict(heldout_features)
        heldout_metrics = direction_metrics(
            heldout_predictions, heldout_targets, model_config["direction_thresholds"]
        )
        print(f"heldout_n={len(heldout_samples)}")
        print(f"heldout_metrics={heldout_metrics}")

    if args.output:
        output_path = resolve_path(args.output, package_root())
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(
            {
                "l2": probe.l2,
                "feature_mean": probe.feature_mean,
                "feature_std": probe.feature_std,
                "target_mean": probe.target_mean,
                "train_features": probe.train_features,
                "alpha": probe.alpha,
                "dino_readout": dino_readout,
                "use_depth": use_depth,
                "depth_grid_size": args.depth_grid_size,
                "direction_thresholds": list(model_config["direction_thresholds"]),
                "model_name": model_config["model_name"],
                "input_size": list(model_config["input_size"]),
                "class_names": list(model_config["class_names"]),
            },
            output_path,
        )
        print(f"saved probe: {output_path}")


if __name__ == "__main__":
    main()
