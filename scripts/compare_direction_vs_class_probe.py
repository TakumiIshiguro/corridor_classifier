#!/usr/bin/env python3
"""Compares two label representations for the ridge linear probe on the
exact same features, samples, and leave-one-session-out folds:

1. direction: independent front/left/right ridge regression (the adopted
   approach), reconstructed into a shape class for class-level scoring.
2. class: one-hot ridge regression directly over the 8 non-turning shape
   classes (what a from-scratch 9-way classifier would reduce to once
   turning is excluded), argmax at prediction time.

Both are scored with the same class-level macro-F1 so the numbers are
directly comparable, answering whether decomposing into open-direction
predictions is actually better than direct shape classification at this
project's data scale -- including for the classes with very few or zero
training examples (cross_road has none at all).
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
    extract_features,
    leave_one_session_out_folds,
)
from corridor_classifier.passage_directions import (
    CLASS_TO_DIRECTIONS,
    class_index_from_directions,
    passage_target_from_index,
    threshold_directions,
)


DEFAULT_L2_CANDIDATES = [0.1, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
SHAPE_CLASSES = list(CLASS_TO_DIRECTIONS)  # 8 non-turning classes, fixed order


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--l2-candidates", type=float, nargs="+", default=DEFAULT_L2_CANDIDATES)
    return parser.parse_args()


@torch.no_grad()
def collect_features_and_labels(dino, loader, device, dino_readout, class_names):
    feature_batches = []
    class_index_batches = []
    for inputs, targets in loader:
        rgb = inputs["rgb"][:, 0].to(device, non_blocking=True)
        features = extract_features(dino, rgb, dino_readout, None, 0)
        feature_batches.append(features.cpu())
        directions = targets["directions"]
        class_indices = torch.tensor(
            [
                SHAPE_CLASSES.index(
                    class_names[class_index_from_directions(row.int().tolist(), class_names)]
                )
                for row in directions
            ]
        )
        class_index_batches.append(class_indices)
    return torch.cat(feature_batches), torch.cat(class_index_batches)


def class_macro_f1(predicted: torch.Tensor, truth: torch.Tensor, num_classes: int) -> float:
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(truth.tolist(), predicted.tolist()):
        confusion[t, p] += 1
    scores = []
    for index in range(num_classes):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        denominator = 2.0 * tp + fp + fn
        scores.append(0.0 if denominator == 0.0 else 2.0 * tp / denominator)
    return sum(scores) / len(scores)


def direction_predictions_to_class_index(direction_scores: torch.Tensor, thresholds, class_names):
    predicted = []
    for row in direction_scores:
        directions = threshold_directions(tuple(row.tolist()), thresholds)
        try:
            class_name = class_names[class_index_from_directions(directions, class_names)]
            predicted.append(SHAPE_CLASSES.index(class_name))
        except (ValueError, KeyError):
            # thresholded combination does not correspond to any known shape
            # (e.g. all-closed except one axis in a way that never occurs in
            # CLASS_TO_DIRECTIONS); count as a fixed wrong-class sentinel.
            predicted.append(-1)
    return torch.tensor(predicted)


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
    class_names = model_config["class_names"]
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

    features, class_indices = collect_features_and_labels(
        dino, loader, device, dino_readout, class_names
    )
    direction_targets = torch.stack(
        [
            passage_target_from_index(
                class_names.index(SHAPE_CLASSES[index]), class_names
            )["directions"]
            for index in class_indices.tolist()
        ]
    )
    class_one_hot = torch.nn.functional.one_hot(
        class_indices, num_classes=len(SHAPE_CLASSES)
    ).float()

    print(f"class distribution: { {name: int((class_indices == i).sum()) for i, name in enumerate(SHAPE_CLASSES)} }")

    direction_scores_by_l2 = {}
    class_scores_by_l2 = {}
    for l2 in args.l2_candidates:
        direction_fold_scores = []
        class_fold_scores = []
        for _, train_idx, held_out_idx in leave_one_session_out_folds(session_names):
            direction_probe = RidgeProbe(l2).fit(
                features[train_idx], direction_targets[train_idx]
            )
            direction_pred_scores = direction_probe.predict(features[held_out_idx])
            direction_pred_class = direction_predictions_to_class_index(
                direction_pred_scores, thresholds, class_names
            )
            valid = direction_pred_class >= 0
            padded_pred = direction_pred_class.clone()
            padded_pred[~valid] = (padded_pred[valid].mode().values if valid.any() else 0)
            direction_fold_scores.append(
                class_macro_f1(padded_pred, class_indices[held_out_idx], len(SHAPE_CLASSES))
            )

            class_probe = RidgeProbe(l2).fit(features[train_idx], class_one_hot[train_idx])
            class_pred_scores = class_probe.predict(features[held_out_idx])
            class_pred = class_pred_scores.argmax(dim=1)
            class_fold_scores.append(
                class_macro_f1(class_pred, class_indices[held_out_idx], len(SHAPE_CLASSES))
            )
        direction_scores_by_l2[l2] = sum(direction_fold_scores) / len(direction_fold_scores)
        class_scores_by_l2[l2] = sum(class_fold_scores) / len(class_fold_scores)

    print("\nLOSO class-level macro-F1 by l2 (direction-decomposition vs direct 8-way class):")
    print(f"{'l2':>10} {'direction->class':>18} {'direct class':>14}")
    for l2 in args.l2_candidates:
        print(f"{l2:>10} {direction_scores_by_l2[l2]:>18.4f} {class_scores_by_l2[l2]:>14.4f}")

    best_direction_l2 = max(direction_scores_by_l2, key=direction_scores_by_l2.get)
    best_class_l2 = max(class_scores_by_l2, key=class_scores_by_l2.get)
    print(f"\nbest direction-decomposition: l2={best_direction_l2} class-macro-F1={direction_scores_by_l2[best_direction_l2]:.4f}")
    print(f"best direct class:            l2={best_class_l2} class-macro-F1={class_scores_by_l2[best_class_l2]:.4f}")


if __name__ == "__main__":
    main()
