#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
import torch
from torch import nn
from torch.utils.data import DataLoader

from corridor_classifier.config import (
    load_training_config,
    package_root,
    resolve_path,
)
from corridor_classifier.dataset import (
    CorridorDataset,
    class_counts,
    load_dataset_samples,
)
from corridor_classifier.dino_classifier import (
    create_dino_model,
    resolve_device,
)
from corridor_classifier.training import (
    MetricsWriter,
    configure_trainable_layers,
    create_optimizer,
    create_scheduler,
    last_blocks_for_epoch,
    run_epoch,
    save_checkpoint,
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
        )

    train_dataset = CorridorDataset(train_samples, model_config["input_size"])
    test_dataset = (
        CorridorDataset(test_samples, model_config["input_size"])
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
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
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
    model = create_dino_model(
        model_name=model_config["model_name"],
        input_size=model_config["input_size"],
        num_classes=model_config["num_classes"],
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
    criterion = nn.CrossEntropyLoss()
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
        f"training samples={len(train_samples)} "
        f"use_test={use_test} test samples={len(test_samples)} device={device}"
    )
    print(
        "train class counts="
        f"{effective_config['dataset']['train_class_counts']}"
    )
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

    best_monitored_loss = float("inf")
    metrics_writer = MetricsWriter(metrics_path)
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

            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                num_classes=model_config["num_classes"],
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=use_amp,
                description=f"train {epoch}/{training['epochs']}",
            )
            test_metrics = (
                run_epoch(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    device=device,
                    num_classes=model_config["num_classes"],
                    use_amp=use_amp,
                    description=f"test {epoch}/{training['epochs']}",
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
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "test_loss": (
                    test_metrics["loss"] if test_metrics is not None else None
                ),
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
            metrics_writer.write(row)
            message = (
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"backbone_lr={learning_rates['backbone']:.3e} "
                f"head_lr={learning_rates['head']:.3e}"
            )
            if test_metrics is not None:
                message += (
                    f" test_loss={test_metrics['loss']:.6f} "
                    f"test_acc={test_metrics['accuracy']:.4f} "
                    f"test_macro_f1={test_metrics['macro_f1']:.4f}"
                )
            print(message)

            monitored_loss = (
                test_metrics["loss"]
                if test_metrics is not None
                else train_metrics["loss"]
            )
            if monitored_loss < best_monitored_loss:
                best_monitored_loss = monitored_loss
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
                print(f"saved best checkpoint: {output_checkpoint}")

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
