#!/usr/bin/env python3
"""Re-picks the best architecture (backbone size, DINO readout, resolution)
using ONLY dataset.train_session_names (a-h), never touching
dataset.test_session_names (i-n). This is necessary once i-n is being kept
as a genuinely unseen sanity check for the deployed model: comparing
several candidate architectures against i-n and picking the best one would
itself be a form of model selection on i-n, defeating the point of holding
it out.

Session "a_f" is a single ~24.5-minute continuous recording (5645 samples,
93% of a-h), so a plain 3-session (a_f, g, h) leave-one-out CV is
unreliable -- the a_f-held-out fold trains on very little data, and the
g/h-held-out folds barely perturb anything. a_f is split by timestamp into
equal-duration chunks and treated as separate pseudo-sessions for LOSO
purposes only (the final fit still uses the real a_f/g/h sessions).
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
from corridor_classifier.linear_probe import direction_metrics, extract_features, leave_one_session_out_folds


DEFAULT_L2_CANDIDATES = [0.1, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]

CANDIDATES = [
    ("vits14_224_regional3", "config/experiments/linear_probe_regional3_depth4", "last_cls_regional3", False),
    ("vitb14_224_regional3", "config/experiments/linear_probe_vitb14_224", "last_cls_regional3", False),
    ("vitb14_224_regional5", "config/experiments/linear_probe_vitb14_224", "last_cls_regional5", False),
    ("vitb14_224_regional3x2", "config/experiments/linear_probe_vitb14_224", "last_cls_regional3x2", False),
    ("vitb14_224_lastcls", "config/experiments/linear_probe_vitb14_224", "last_cls", False),
    ("vitl14_224_regional3", "config/experiments/linear_probe_vitl14_224", "last_cls_regional3", False),
    ("vitb14_336_regional3", "config/experiments/linear_probe_vitb14_336", "last_cls_regional3", False),
    ("vitb14_224_regional5+depth4", "config/experiments/linear_probe_vitb14_224", "last_cls_regional5", True),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-candidates", type=float, nargs="+", default=DEFAULT_L2_CANDIDATES)
    parser.add_argument("--pseudo-session-chunks", type=int, default=6)
    parser.add_argument(
        "--only", nargs="+", default=None, help="Restrict to these candidate names."
    )
    return parser.parse_args()


def chunked_pseudo_sessions(session_names, stamps, chunked_session, num_chunks):
    """Replaces session_names entries equal to chunked_session with
    "{chunked_session}_{k}" based on equal time-duration chunks, leaving
    every other session name untouched.
    """
    indices = [i for i, name in enumerate(session_names) if name == chunked_session]
    if not indices:
        return list(session_names)
    chunk_stamps = [stamps[i] for i in indices]
    minimum, maximum = min(chunk_stamps), max(chunk_stamps)
    span = max(maximum - minimum, 1e-6)
    result = list(session_names)
    for i in indices:
        fraction = (stamps[i] - minimum) / span
        chunk_index = min(int(fraction * num_chunks), num_chunks - 1)
        result[i] = f"{chunked_session}_{chunk_index}"
    return result


@torch.no_grad()
def collect_rgb_features_and_targets(dino, loader, device, dino_readout, use_depth, depth_grid_size=4):
    feature_batches, target_batches = [], []
    for inputs, targets in loader:
        rgb = inputs["rgb"][:, 0].to(device, non_blocking=True)
        depth = inputs["depth"][:, 0].to(device, non_blocking=True) if use_depth else None
        feature_batches.append(
            extract_features(dino, rgb, dino_readout, depth, depth_grid_size).cpu()
        )
        target_batches.append(targets["directions"])
    return torch.cat(feature_batches), torch.cat(target_batches)


def run_loso_eigh(features, targets, session_names, thresholds, candidates):
    scores_by_l2 = {l2: {} for l2 in candidates}
    for held_out_name, train_idx, held_out_idx in leave_one_session_out_folds(session_names):
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
            metrics = direction_metrics(predictions.float(), held_targets, thresholds)
            scores_by_l2[l2][held_out_name] = metrics["direction_macro_f1"]

    # a_f_* folds hold out a time chunk of the same continuous recording
    # (near-identical environment interpolation); g/h folds hold out a
    # genuinely different session (the real cross-environment signal).
    # Select l2 on the full blended average, but also report the
    # session-only average since it is less likely to be inflated by
    # interpolation.
    mean_scores = {
        l2: sum(per_fold.values()) / len(per_fold) for l2, per_fold in scores_by_l2.items()
    }
    session_only_scores = {
        l2: sum(v for name, v in per_fold.items() if not name.startswith("a_f_"))
        / max(1, sum(1 for name in per_fold if not name.startswith("a_f_")))
        for l2, per_fold in scores_by_l2.items()
    }
    best_l2 = max(mean_scores, key=mean_scores.get)
    return best_l2, mean_scores[best_l2], session_only_scores[best_l2]


def evaluate_candidate(name, config_dir, dino_readout, use_depth, args, dino_cache):
    config = load_training_config(config_dir)
    model_config = config["model"]
    dataset_config = config["dataset"]

    model_name = model_config["model_name"]
    input_size = tuple(model_config["input_size"])
    cache_key = (model_name, input_size)
    if cache_key not in dino_cache:
        training_config = config["training"]
        device = resolve_device(model_config.get("device", "auto"))
        pretrained_path_config = str(training_config.get("pretrained_weights_path", "")).strip()
        pretrained_path = (
            resolve_path(pretrained_path_config, package_root()) if pretrained_path_config else None
        )
        dino = create_dino_model(
            model_name=model_name,
            input_size=list(input_size),
            num_classes=0,
            pretrained=True,
            pretrained_weights_path=pretrained_path,
        ).to(device)
        dino.eval()
        dino_cache[cache_key] = (dino, device)
    dino, device = dino_cache[cache_key]

    model_config["use_depth"] = use_depth
    train_samples = load_dataset_samples(
        resolve_path(dataset_config["train_data_dir"], package_root()),
        model_config["num_classes"],
        dataset_config["train_session_names"],
    )
    dataset = CorridorMultiInputDataset(
        train_samples, model_config["input_size"], model_config, sequence_step=1
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
    )
    session_names = [sequence[-1].session_name for sequence in dataset.sequences]
    stamps = [sequence[-1].stamp for sequence in dataset.sequences]
    pseudo_sessions = chunked_pseudo_sessions(
        session_names, stamps, "a_f", args.pseudo_session_chunks
    )

    features, targets = collect_rgb_features_and_targets(dino, loader, device, dino_readout, use_depth)
    best_l2, blended_score, session_only_score = run_loso_eigh(
        features, targets, pseudo_sessions, model_config["direction_thresholds"], args.l2_candidates
    )
    print(
        f"{name:>28}  dim={features.shape[1]:>5}  l2={best_l2:>8}  "
        f"blended-F1={blended_score:.4f}  session-only-F1(g/h)={session_only_score:.4f}"
    )
    return name, best_l2, blended_score, session_only_score


def main():
    args = parse_args()
    dino_cache = {}
    results = []
    candidates = (
        CANDIDATES
        if not args.only
        else [c for c in CANDIDATES if c[0] in args.only]
    )
    for name, config_dir, dino_readout, use_depth in candidates:
        try:
            results.append(
                evaluate_candidate(name, config_dir, dino_readout, use_depth, args, dino_cache)
            )
        except Exception as error:
            print(f"{name:>28}  FAILED: {error}")
    print(
        "\nRanking by session-only-F1 (g/h held out; the genuine "
        "cross-environment signal, a_f time-chunk folds excluded):"
    )
    for name, l2, blended, session_only in sorted(results, key=lambda r: -r[3]):
        print(f"  {name:>28}  l2={l2:>8}  session-only-F1={session_only:.4f}  blended-F1={blended:.4f}")
    best = max(results, key=lambda r: r[3])
    print(f"\nbest by session-only-F1: {best[0]} l2={best[1]} session-only-F1={best[3]:.4f}")


if __name__ == "__main__":
    main()
