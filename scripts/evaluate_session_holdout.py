#!/usr/bin/env python3
"""Evaluates a trained passage_directions checkpoint on dataset sessions that
were never used for training or per-epoch checkpoint selection, to get an
untouched estimate of generalization instead of the epoch-selection-biased
number reported during training.
"""
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
from corridor_classifier.training import PassageDirectionLoss, run_passage_epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--session-names",
        required=True,
        nargs="+",
        help="Dataset session names that were held out of training and "
        "validation entirely.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Dataset directory containing the held-out sessions. Defaults "
        "to dataset.test_data_dir from the config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_training_config(args.config_dir)
    model_config = config["model"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    if model_config.get("output_mode") != "passage_directions":
        raise ValueError("holdout evaluation requires passage_directions output mode")

    data_dir = args.data_dir or dataset_config["test_data_dir"]
    samples = load_dataset_samples(
        resolve_path(data_dir, package_root()),
        model_config["num_classes"],
        args.session_names,
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

    metrics = run_passage_epoch(
        model=model,
        loader=loader,
        criterion=PassageDirectionLoss().to(device),
        device=device,
        direction_thresholds=model_config["direction_thresholds"],
        use_amp=False,
        description="holdout",
    )
    print(f"checkpoint={checkpoint}")
    print(f"holdout_sessions={args.session_names} samples={len(dataset)}")
    for key, value in metrics.items():
        print(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")


if __name__ == "__main__":
    main()
