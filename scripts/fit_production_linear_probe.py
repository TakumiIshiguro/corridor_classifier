#!/usr/bin/env python3
"""Fits the final production ridge linear-probe checkpoint.

By default this uses every labeled session (train + test) for both
leave-one-session-out (LOSO) regularization selection and the final fit,
on the view that once a method is validated, a deployed checkpoint should
not waste any of this project's limited data.

Pass --exclude-test-sessions to instead fit (and run LOSO CV) using only
dataset.train_session_names, leaving dataset.test_session_names completely
untouched by the checkpoint -- so there remains a genuinely unseen set of
sessions to sanity-check the deployed model against later. Add
--heldout-eval to also run that one-time check immediately and record it
in the checkpoint.
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
from corridor_classifier.linear_probe import (
    RidgeProbe,
    direction_metrics,
    extract_features,
    select_l2_by_leave_one_session_out,
)


DEFAULT_L2_CANDIDATES = [0.1, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--depth-grid-size", type=int, default=4)
    parser.add_argument("--l2-candidates", type=float, nargs="+", default=DEFAULT_L2_CANDIDATES)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude-test-sessions",
        action="store_true",
        help="Fit only on dataset.train_session_names (LOSO CV also runs only "
        "across those sessions). dataset.test_session_names are excluded "
        "entirely from fitting and regularization selection, so they remain "
        "genuinely unseen by the saved checkpoint.",
    )
    parser.add_argument(
        "--heldout-eval",
        action="store_true",
        help="After fitting, evaluate once on dataset.test_session_names "
        "(only meaningful together with --exclude-test-sessions).",
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    config = load_training_config(args.config_dir)
    model_config = config["model"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    if model_config.get("output_mode") != "passage_directions":
        raise ValueError("linear probe fitting requires passage_directions output mode")
    if int(model_config.get("sequence_length", 1)) != 1:
        raise ValueError("linear probe fitting requires model.sequence_length == 1")

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
    use_depth = bool(model_config["use_depth"]) and not args.no_depth
    model_config["use_depth"] = use_depth
    batch_size = int(training_config["batch_size"])

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
    all_samples = list(train_samples) if args.exclude_test_sessions else (
        list(train_samples) + list(test_samples)
    )
    all_session_names = sorted({sample.session_name for sample in all_samples})
    print(f"fitting on: {all_session_names} ({len(all_samples)} raw samples)")
    if args.exclude_test_sessions:
        held_out_names = sorted({sample.session_name for sample in test_samples})
        print(
            f"held out entirely from fitting/regularization selection: {held_out_names}"
        )

    dataset = CorridorMultiInputDataset(
        all_samples,
        model_config["input_size"],
        model_config,
        sequence_step=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
    )
    session_names_per_sequence = [
        sequence[-1].session_name for sequence in dataset.sequences
    ]
    print(f"extracting features: {len(dataset)} sequences (turning frames already excluded)")
    features, targets = collect_features(
        dino, loader, device, dino_readout, use_depth, args.depth_grid_size
    )

    best_l2, scores_by_l2 = select_l2_by_leave_one_session_out(
        features, targets, session_names_per_sequence,
        model_config["direction_thresholds"], args.l2_candidates,
    )
    print("leave-one-session-out mean direction macro-F1 by l2:")
    for l2 in args.l2_candidates:
        marker = " <-- selected" if l2 == best_l2 else ""
        print(f"  l2={l2}: {scores_by_l2[l2]:.4f}{marker}")
    print(
        "This LOSO score is the cross-validated estimate of how the final "
        "probe (fit below on every session) should generalize."
    )

    final_probe = RidgeProbe(best_l2).fit(features, targets)
    training_fit_metrics = direction_metrics(
        final_probe.predict(features), targets, model_config["direction_thresholds"]
    )
    print(f"final probe metrics on its own training data (optimistic, for sanity only): {training_fit_metrics}")

    heldout_metrics = None
    if args.heldout_eval:
        if not args.exclude_test_sessions:
            raise ValueError("--heldout-eval requires --exclude-test-sessions")
        heldout_dataset = CorridorMultiInputDataset(
            test_samples, model_config["input_size"], model_config, sequence_step=1
        )
        heldout_loader = DataLoader(
            heldout_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataset_config["num_workers"],
            pin_memory=True,
        )
        heldout_features, heldout_targets = collect_features(
            dino, heldout_loader, device, dino_readout, use_depth, args.depth_grid_size
        )
        heldout_metrics = direction_metrics(
            final_probe.predict(heldout_features),
            heldout_targets,
            model_config["direction_thresholds"],
        )
        heldout_session_names = sorted({s.session_name for s in test_samples})
        print(f"untouched held-out evaluation on {heldout_session_names}: {heldout_metrics}")

    output_path = resolve_path(args.output, package_root())
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(
        {
            "l2": final_probe.l2,
            "feature_mean": final_probe.feature_mean,
            "feature_std": final_probe.feature_std,
            "target_mean": final_probe.target_mean,
            "train_features": final_probe.train_features,
            "alpha": final_probe.alpha,
            "dino_readout": dino_readout,
            "use_depth": use_depth,
            "depth_grid_size": args.depth_grid_size,
            "depth_min_m": float(model_config.get("depth_min_m", 0.1)),
            "depth_max_m": float(model_config.get("depth_max_m", 10.0)),
            "direction_thresholds": list(model_config["direction_thresholds"]),
            "model_name": model_config["model_name"],
            "input_size": list(model_config["input_size"]),
            "class_names": list(model_config["class_names"]),
            "pretrained_weights_path": pretrained_path,
            "loso_mean_direction_macro_f1": scores_by_l2[best_l2],
            "fitted_sessions": all_session_names,
            "heldout_sessions": (
                sorted({s.session_name for s in test_samples})
                if args.exclude_test_sessions
                else []
            ),
            "heldout_metrics": heldout_metrics,
        },
        output_path,
    )
    print(f"saved production probe: {output_path}")


if __name__ == "__main__":
    main()
