#!/usr/bin/env python3
"""Tests whether simple post-hoc temporal aggregation (no learned depth, no
learned recurrence) over the already-fitted single-frame ridge probe closes
the gap to the GRU+depth result on i-n, using the exact same 3-frame /
1-second-spacing window the GRU uses (sequence_length=3, frame_stride=4).

Three variants, all built from the SAME single-frame ridge-probe
predictions:
  - last-frame-only: the probe's normal single-frame prediction (no
    temporal use at all), evaluated on windows requiring 3 valid frames of
    context so the sample set matches the other two variants exactly.
  - majority-vote: threshold each of the 3 frames independently, then take
    the majority (>=2 of 3) open/closed vote per direction -- what was
    asked about.
  - mean-then-threshold: average the 3 frames' continuous ridge scores,
    then threshold once.
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
from corridor_classifier.linear_probe import RidgeProbe, direction_metrics, extract_features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=4)
    return parser.parse_args()


def load_probe(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    probe = RidgeProbe(float(checkpoint["l2"]))
    probe.feature_mean = checkpoint["feature_mean"].to(device)
    probe.feature_std = checkpoint["feature_std"].to(device)
    probe.target_mean = checkpoint["target_mean"].to(device)
    probe.train_features = checkpoint["train_features"].to(device)
    probe.alpha = checkpoint["alpha"].to(device)
    return probe, checkpoint


@torch.no_grad()
def collect_windowed_scores(dino, probe, loader, device, dino_readout, sequence_length):
    """Returns (num_sequences, sequence_length, 3) continuous ridge scores
    for every frame in every temporal window, plus the (N, 3) direction
    targets for the window's last frame.
    """
    score_batches, target_batches = [], []
    for inputs, targets in loader:
        rgb = inputs["rgb"].to(device, non_blocking=True)  # (B, T, C, H, W)
        batch_size, seq_len = rgb.shape[:2]
        flat_rgb = rgb.flatten(0, 1)
        features = extract_features(dino, flat_rgb, dino_readout, None, 0)
        scores = probe.predict(features).reshape(batch_size, seq_len, 3)
        score_batches.append(scores.cpu())
        target_batches.append(targets["directions"])
    return torch.cat(score_batches), torch.cat(target_batches)


def main():
    args = parse_args()
    config = load_training_config(args.config_dir)
    model_config = config["model"]
    dataset_config = config["dataset"]
    training_config = config["training"]

    model_config["use_depth"] = False
    model_config["sequence_length"] = args.sequence_length
    model_config["frame_stride"] = args.frame_stride
    model_config["use_gru"] = True  # only to satisfy config's sequence_length>1 check

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
    thresholds = torch.as_tensor(model_config["direction_thresholds"])

    probe, probe_checkpoint = load_probe(
        resolve_path(args.probe_checkpoint, package_root()), device
    )
    if probe_checkpoint["dino_readout"] != dino_readout:
        raise ValueError(
            f"config dino_readout={dino_readout} does not match probe "
            f"checkpoint's {probe_checkpoint['dino_readout']}"
        )

    test_samples = load_dataset_samples(
        resolve_path(dataset_config["test_data_dir"], package_root()),
        model_config["num_classes"],
        dataset_config["test_session_names"],
    )
    dataset = CorridorMultiInputDataset(
        test_samples, model_config["input_size"], model_config, sequence_step=1
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
    )
    print(f"i-n windows: {len(dataset)} (sequence_length={args.sequence_length}, frame_stride={args.frame_stride})")

    scores, targets = collect_windowed_scores(
        dino, probe, loader, device, dino_readout, args.sequence_length
    )  # scores: (N, T, 3)

    last_frame_scores = scores[:, -1]
    last_frame_metrics = direction_metrics(last_frame_scores, targets, thresholds)
    print(f"last-frame-only:      {last_frame_metrics}")

    per_frame_open = scores >= thresholds.view(1, 1, 3)
    vote_open = per_frame_open.sum(dim=1) > (scores.shape[1] // 2)
    vote_scores = vote_open.float()
    vote_metrics = direction_metrics(vote_scores, targets, [0.5, 0.5, 0.5])
    print(f"majority-vote (>={scores.shape[1] // 2 + 1}/{scores.shape[1]}): {vote_metrics}")

    mean_scores = scores.mean(dim=1)
    mean_metrics = direction_metrics(mean_scores, targets, thresholds)
    print(f"mean-then-threshold:  {mean_metrics}")


if __name__ == "__main__":
    main()
