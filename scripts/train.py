#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from corridor_classifier.config import (
    load_training_config,
    package_root,
    resolve_path,
)
from corridor_classifier.dataset import (
    CorridorMultiInputDataset,
    class_counts,
    load_dataset_samples,
)
from corridor_classifier.dino_classifier import resolve_device
from corridor_classifier.models import create_corridor_model
from corridor_classifier.passage_directions import (
    inverse_frequency_positive_weights,
    passage_label_counts,
)
from corridor_classifier.training import (
    MetricsWriter,
    PassageDirectionLoss,
    class_weights_from_counts,
    configure_trainable_layers,
    create_optimizer,
    create_scheduler,
    last_blocks_for_epoch,
    passage_direction_sampling_weights,
    run_epoch,
    run_passage_epoch,
    save_checkpoint,
    sequence_sampling_weights,
    set_random_seed,
    write_effective_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    if argv is None:
        argv = sys.argv
    return parser.parse_args(rospy.myargv(argv=argv)[1:])


def main():
    args = parse_args()
    config = load_training_config(args.config_dir)
    model_config = config["model"]
    dataset_config = config["dataset"]
    training = config["training"]
    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]
    set_random_seed(training["seed"])

    train_data_dir = resolve_path(
        dataset_config["train_data_dir"],
        package_root(),
    )
    train_samples = load_dataset_samples(
        dataset_dir=train_data_dir,
        num_classes=model_config["num_classes"],
        session_names=dataset_config["train_session_names"],
    )
    use_test = bool(training["use_test"])
    test_data_dir = ""
    test_samples = []
    if use_test:
        test_data_dir = resolve_path(
            dataset_config["test_data_dir"],
            package_root(),
        )
        test_samples = load_dataset_samples(
            dataset_dir=test_data_dir,
            num_classes=model_config["num_classes"],
            session_names=dataset_config["test_session_names"],
        )

    train_dataset = CorridorMultiInputDataset(
        train_samples,
        model_config["input_size"],
        model_config,
        augmentation_config=training.get("augmentation"),
        sequence_step=dataset_config.get("train_sequence_step", 1),
    )
    test_dataset = (
        CorridorMultiInputDataset(
            test_samples,
            model_config["input_size"],
            model_config,
            sequence_step=dataset_config.get("test_sequence_step", 1),
        )
        if use_test
        else None
    )

    device = resolve_device(model_config.get("device", "auto"))
    use_amp = bool(training["use_amp"]) and device.type == "cuda"
    loader_kwargs = {
        "batch_size": training["batch_size"],
        "num_workers": dataset_config["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(training["seed"])
    sampling_strategy = str(training.get("sampling_strategy", "shuffle"))
    sampler = None
    if sampling_strategy == "passage_direction_inverse_sqrt":
        sampling_weights = passage_direction_sampling_weights(
            [sequence[-1].class_index for sequence in train_dataset.sequences],
            model_config["class_names"],
            model_config["turning_class_name"],
            maximum_direction_factor=float(
                training.get("maximum_sampling_direction_factor", 2.0)
            ),
        )
        sampler = WeightedRandomSampler(
            sampling_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
    elif sampling_strategy == "class_session_inverse_sqrt":
        sampling_weights = sequence_sampling_weights(
            [sequence[-1].class_index for sequence in train_dataset.sequences],
            [sequence[-1].session_name for sequence in train_dataset.sequences],
            maximum_class_factor=float(
                training.get("maximum_sampling_class_factor", 4.0)
            ),
            maximum_session_factor=float(
                training.get("maximum_sampling_session_factor", 4.0)
            ),
        )
        sampler = WeightedRandomSampler(
            sampling_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
    elif sampling_strategy != "shuffle":
        raise ValueError(f"unsupported sampling strategy: {sampling_strategy}")
    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        **loader_kwargs,
    )
    test_loader = (
        DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        if test_dataset is not None
        else None
    )

    pretrained_path = resolve_path(
        training["pretrained_weights_path"],
        package_root(),
    )
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"DINOv2 pretrained weights were not found: {pretrained_path}"
        )
    model = create_corridor_model(
        model_config=model_config,
        pretrained=True,
        pretrained_weights_path=pretrained_path,
    )
    model.to(device)
    optimizer = create_optimizer(model, optimizer_config)
    scheduler = create_scheduler(
        optimizer=optimizer,
        config=scheduler_config,
        total_epochs=training["epochs"],
        steps_per_epoch=len(train_loader),
    )
    train_sequence_counts = class_counts(
        [sequence[-1] for sequence in train_dataset.sequences],
        model_config["num_classes"],
    )
    output_mode = str(model_config.get("output_mode", "class"))
    passage_counts = None
    if output_mode == "passage_directions":
        passage_counts = passage_label_counts(
            [sequence[-1].class_index for sequence in train_dataset.sequences],
            model_config["class_names"],
            model_config["turning_class_name"],
        )
        maximum_positive_weight = float(
            training.get("maximum_positive_weight", 5.0)
        )
        maximum_direction_positive_weight = float(
            training.get(
                "maximum_direction_positive_weight",
                maximum_positive_weight,
            )
        )
        maximum_turning_positive_weight = float(
            training.get(
                "maximum_turning_positive_weight",
                maximum_positive_weight,
            )
        )
        direction_pos_weight = inverse_frequency_positive_weights(
            passage_counts["direction_positive"],
            passage_counts["direction_negative"],
            maximum_direction_positive_weight,
        ).to(device)
        turning_pos_weight = inverse_frequency_positive_weights(
            [passage_counts["turning_positive"]],
            [passage_counts["turning_negative"]],
            maximum_turning_positive_weight,
        ).to(device)[0]
        criterion = PassageDirectionLoss(
            direction_pos_weight=direction_pos_weight,
            turning_pos_weight=turning_pos_weight,
            direction_loss_weight=float(
                training.get("direction_loss_weight", 1.0)
            ),
            turning_loss_weight=float(
                training.get("turning_loss_weight", 1.0)
            ),
        ).to(device)
        evaluation_criterion = PassageDirectionLoss().to(device)
        class_weights = None
    else:
        class_weights = class_weights_from_counts(
            train_sequence_counts,
            method=training.get("class_weighting", "none"),
            maximum_weight=float(training.get("maximum_class_weight", 4.0)),
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        evaluation_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_checkpoint = resolve_path(
        training["output_checkpoint"],
        package_root(),
    )
    final_checkpoint = resolve_path(
        training["final_checkpoint"],
        package_root(),
    )
    metrics_path = resolve_path(training["metrics_path"], package_root())
    effective_config = {
        "model": model_config,
        "dataset": {
            **dataset_config,
            "train_data_dir": train_data_dir,
            "test_data_dir": test_data_dir,
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "train_class_counts": class_counts(
                train_samples,
                model_config["num_classes"],
            ),
            "test_class_counts": (
                class_counts(test_samples, model_config["num_classes"])
                if use_test
                else []
            ),
            "train_passage_label_counts": passage_counts,
        },
        "training": training,
        "optimizer": optimizer_config,
        "scheduler": scheduler_config,
        "device": str(device),
        "pretrained_weights_path": pretrained_path,
    }
    write_effective_config(
        os.path.splitext(output_checkpoint)[0] + ".yaml",
        effective_config,
    )

    print(
        f"architecture={model_config['architecture']} "
        f"training sequences={len(train_dataset)} raw samples={len(train_samples)} "
        f"use_test={use_test} test samples={len(test_samples)} device={device}"
    )
    print(
        "train class counts="
        f"{effective_config['dataset']['train_class_counts']}"
    )
    print(f"train sequence class counts={train_sequence_counts}")
    if output_mode == "passage_directions":
        print(f"passage label counts={passage_counts}")
        print(
            "positive weights="
            f"directions={direction_pos_weight.cpu().tolist()} "
            f"turning={float(turning_pos_weight.cpu()):.4f}"
        )
    else:
        print(f"loss class weights={class_weights.cpu().tolist()}")
    print(f"sampling strategy={sampling_strategy}")
    if use_test:
        print(
            "test class counts="
            f"{effective_config['dataset']['test_class_counts']}"
        )
    missing_classes = [
        model_config["class_names"][index]
        for index, count in enumerate(
            effective_config["dataset"]["train_class_counts"]
        )
        if count == 0
    ]
    if missing_classes:
        print(
            "WARNING: training dataset has no samples for class(es): "
            + ", ".join(missing_classes)
        )

    checkpoint_metric = str(
        training.get(
            "checkpoint_metric",
            (
                "test_direction_macro_f1"
                if use_test and output_mode == "passage_directions"
                else "train_direction_macro_f1"
                if output_mode == "passage_directions"
                else "test_macro_f1"
                if use_test
                else "train_macro_f1"
            ),
        )
    )
    maximize_checkpoint_metric = not checkpoint_metric.endswith("loss")
    best_monitored_value = (
        float("-inf") if maximize_checkpoint_metric else float("inf")
    )
    metrics_writer = MetricsWriter(
        metrics_path,
        fields=(
            MetricsWriter.PASSAGE_FIELDS
            if output_mode == "passage_directions"
            else MetricsWriter.CLASSIFICATION_FIELDS
        ),
    )
    try:
        current_last_blocks = None
        for epoch in range(1, training["epochs"] + 1):
            last_blocks = last_blocks_for_epoch(
                epoch,
                training["freeze_backbone_epochs"],
                training["unfreeze_schedule"],
            )
            if last_blocks != current_last_blocks:
                trainable_info = configure_trainable_layers(
                    model,
                    last_blocks,
                )
                current_last_blocks = last_blocks
                print(
                    f"epoch={epoch} unfreeze_last_blocks={last_blocks} "
                    f"trainable={trainable_info['trainable_parameters']}/"
                    f"{trainable_info['total_parameters']}"
                )

            common_train_arguments = {
                "model": model,
                "loader": train_loader,
                "criterion": criterion,
                "device": device,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "scaler": scaler,
                "use_amp": use_amp,
                "description": f"train {epoch}/{training['epochs']}",
            }
            if output_mode == "passage_directions":
                train_metrics = run_passage_epoch(
                    direction_thresholds=model_config[
                        "direction_thresholds"
                    ],
                    turning_threshold=model_config["turning_threshold"],
                    **common_train_arguments,
                )
            else:
                train_metrics = run_epoch(
                    num_classes=model_config["num_classes"],
                    **common_train_arguments,
                )
            test_metrics = (
                (
                    run_passage_epoch(
                        model=model,
                        loader=test_loader,
                        criterion=evaluation_criterion,
                        device=device,
                        direction_thresholds=model_config[
                            "direction_thresholds"
                        ],
                        turning_threshold=model_config[
                            "turning_threshold"
                        ],
                        use_amp=use_amp,
                        description=f"test {epoch}/{training['epochs']}",
                    )
                    if output_mode == "passage_directions"
                    else run_epoch(
                        model=model,
                        loader=test_loader,
                        criterion=evaluation_criterion,
                        device=device,
                        num_classes=model_config["num_classes"],
                        use_amp=use_amp,
                        description=f"test {epoch}/{training['epochs']}",
                    )
                )
                if test_loader is not None
                else None
            )
            learning_rates = {
                group["name"]: float(group["lr"])
                for group in optimizer.param_groups
            }

            row = {
                "epoch": epoch,
                "last_blocks": last_blocks,
                "trainable_parameters": trainable_info[
                    "trainable_parameters"
                ],
                "backbone_learning_rate": learning_rates["backbone"],
                "head_learning_rate": learning_rates["head"],
                "train_loss": train_metrics["loss"],
                "test_loss": (
                    test_metrics["loss"] if test_metrics is not None else None
                ),
            }
            if output_mode == "passage_directions":
                row.update(
                    {
                        f"train_{key}": value
                        for key, value in train_metrics.items()
                        if key != "loss"
                    }
                )
                if test_metrics is not None:
                    row.update(
                        {
                            f"test_{key}": value
                            for key, value in test_metrics.items()
                            if key != "loss"
                        }
                    )
            else:
                row.update(
                    {
                        "train_accuracy": train_metrics["accuracy"],
                        "train_macro_f1": train_metrics["macro_f1"],
                        "test_accuracy": (
                            test_metrics["accuracy"]
                            if test_metrics is not None
                            else None
                        ),
                        "test_macro_f1": (
                            test_metrics["macro_f1"]
                            if test_metrics is not None
                            else None
                        ),
                    }
                )
            metrics_writer.write(row)
            message = f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            if output_mode == "passage_directions":
                message += (
                    f"train_dir_f1={train_metrics['direction_macro_f1']:.4f} "
                    f"train_turn_f1={train_metrics['turning_f1']:.4f} "
                )
            else:
                message += f"train_acc={train_metrics['accuracy']:.4f} "
            message += (
                f"backbone_lr={learning_rates['backbone']:.3e} "
                f"head_lr={learning_rates['head']:.3e}"
            )
            if test_metrics is not None:
                message += f" test_loss={test_metrics['loss']:.6f} "
                if output_mode == "passage_directions":
                    message += (
                        f"test_dir_f1={test_metrics['direction_macro_f1']:.4f} "
                        f"test_turn_f1={test_metrics['turning_f1']:.4f}"
                    )
                else:
                    message += (
                        f"test_acc={test_metrics['accuracy']:.4f} "
                        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
                    )
            print(message)

            metric_source, metric_name = checkpoint_metric.split("_", 1)
            monitored_metrics = (
                test_metrics if metric_source == "test" else train_metrics
            )
            if monitored_metrics is None or metric_name not in monitored_metrics:
                raise ValueError(
                    f"checkpoint metric is unavailable: {checkpoint_metric}"
                )
            monitored_value = float(monitored_metrics[metric_name])
            improved = (
                monitored_value > best_monitored_value
                if maximize_checkpoint_metric
                else monitored_value < best_monitored_value
            )
            if improved:
                best_monitored_value = monitored_value
                save_checkpoint(
                    output_checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    model_config,
                    training,
                    optimizer_config,
                    scheduler_config,
                    row,
                )
                print(
                    f"saved best checkpoint: {output_checkpoint} "
                    f"({checkpoint_metric}={monitored_value:.6f})"
                )

        save_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            scheduler,
            training["epochs"],
            model_config,
            training,
            optimizer_config,
            scheduler_config,
            row,
        )
        print(f"saved final checkpoint: {final_checkpoint}")
    finally:
        metrics_writer.close()


if __name__ == "__main__":
    main()
