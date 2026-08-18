#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch
from torch.utils.data import DataLoader

from corridor_classifier.config import load_training_config, package_root, resolve_path
from corridor_classifier.dataset import CorridorMultiInputDataset, load_dataset_samples
from corridor_classifier.dino_classifier import resolve_device
from corridor_classifier.models import create_corridor_model, load_model_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


def binary_f1(predictions, targets):
    true_positive = int((predictions & targets).sum())
    false_positive = int((predictions & ~targets).sum())
    false_negative = int((~predictions & targets).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def metrics(direction_probabilities, direction_targets, turning_probabilities, turning_targets, thresholds):
    direction_predictions = direction_probabilities >= thresholds[:3]
    direction_f1 = [
        binary_f1(direction_predictions[:, index], direction_targets[:, index])
        for index in range(3)
    ]
    turning_predictions = turning_probabilities >= thresholds[3]
    return {
        "front_f1": direction_f1[0],
        "left_f1": direction_f1[1],
        "right_f1": direction_f1[2],
        "direction_macro_f1": sum(direction_f1) / 3.0,
        "direction_exact_accuracy": float(
            (direction_predictions == direction_targets).all(dim=1).float().mean()
        ),
        "turning_f1": binary_f1(turning_predictions, turning_targets),
        "turning_accuracy": float(
            (turning_predictions == turning_targets).float().mean()
        ),
    }


def best_threshold(probabilities, targets, candidates):
    scored = [
        (binary_f1(probabilities >= threshold, targets), threshold)
        for threshold in candidates
    ]
    return max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))


def main():
    args = parse_args()
    config = load_training_config(args.config_dir)
    model_config = config["model"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    if model_config.get("output_mode") != "passage_directions":
        raise ValueError("threshold tuning requires passage_directions output mode")

    samples = load_dataset_samples(
        resolve_path(dataset_config["test_data_dir"], package_root()),
        model_config["num_classes"],
        dataset_config["test_session_names"],
    )
    dataset = CorridorMultiInputDataset(
        samples,
        model_config["input_size"],
        model_config,
        sequence_step=dataset_config.get("test_sequence_step", 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
    )
    device = resolve_device(model_config.get("device", "auto"))
    checkpoint = args.checkpoint or model_config["checkpoint_path"]
    checkpoint = resolve_path(checkpoint, package_root())
    model = create_corridor_model(model_config).to(device)
    load_model_checkpoint(model, model_config, checkpoint)
    model.eval()

    direction_probabilities = []
    direction_targets = []
    turning_probabilities = []
    turning_targets = []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(inputs)
            mask = targets["direction_mask"].bool()
            direction_probabilities.append(torch.sigmoid(outputs["direction_logits"]).cpu()[mask])
            direction_targets.append(targets["directions"].bool()[mask])
            turning_probabilities.append(torch.sigmoid(outputs["turning_logits"]).cpu())
            turning_targets.append(targets["turning"].bool())

    direction_probabilities = torch.cat(direction_probabilities)
    direction_targets = torch.cat(direction_targets)
    turning_probabilities = torch.cat(turning_probabilities)
    turning_targets = torch.cat(turning_targets)
    step = float(args.threshold_step)
    if not 0.0 < step < 1.0:
        raise ValueError("threshold-step must be between 0 and 1")
    candidates = [step * index for index in range(1, int(1.0 / step))]
    tuned = [
        best_threshold(direction_probabilities[:, index], direction_targets[:, index], candidates)
        for index in range(3)
    ]
    tuned.append(best_threshold(turning_probabilities, turning_targets, candidates))
    thresholds = torch.tensor([item[1] for item in tuned])
    default_thresholds = torch.tensor(
        list(model_config["direction_thresholds"]) + [model_config["turning_threshold"]]
    )
    print(f"checkpoint={checkpoint}")
    print(f"samples=directions:{len(direction_targets)} turning:{len(turning_targets)}")
    print(f"default_thresholds={default_thresholds.tolist()}")
    print(f"default_metrics={metrics(direction_probabilities, direction_targets, turning_probabilities, turning_targets, default_thresholds)}")
    print(f"tuned_thresholds={thresholds.tolist()}")
    print(f"tuned_metrics={metrics(direction_probabilities, direction_targets, turning_probabilities, turning_targets, thresholds)}")


if __name__ == "__main__":
    main()
