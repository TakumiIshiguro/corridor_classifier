import csv
import os
import random
from collections import Counter
from contextlib import nullcontext
from math import cos, pi
from typing import Dict, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from corridor_classifier.passage_directions import (
    passage_label_counts,
    passage_target_from_index,
)


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def class_weights_from_counts(
    counts: Sequence[int],
    method: str = "none",
    maximum_weight: float = 4.0,
) -> torch.Tensor:
    method = str(method).strip().lower()
    values = torch.as_tensor(counts, dtype=torch.float32)
    positive = values > 0
    weights = torch.zeros_like(values)
    if method == "none":
        weights[positive] = 1.0
        return weights
    if method != "inverse_sqrt":
        raise ValueError("class weighting must be none or inverse_sqrt")
    if positive.any():
        weights[positive] = torch.sqrt(values[positive].max() / values[positive])
        weights.clamp_(max=float(maximum_weight))
        weights[positive] /= weights[positive].mean()
    return weights


def sequence_sampling_weights(
    labels: Sequence[int],
    sessions: Sequence[str],
    maximum_class_factor: float = 4.0,
    maximum_session_factor: float = 4.0,
) -> torch.Tensor:
    if len(labels) != len(sessions) or not labels:
        raise ValueError("labels and sessions must have the same non-zero length")
    label_counts = Counter(int(label) for label in labels)
    session_counts = Counter(str(session) for session in sessions)
    largest_class = max(label_counts.values())
    largest_session = max(session_counts.values())
    weights = []
    for label, session in zip(labels, sessions):
        class_factor = min(
            (largest_class / label_counts[int(label)]) ** 0.5,
            float(maximum_class_factor),
        )
        session_factor = min(
            (largest_session / session_counts[str(session)]) ** 0.5,
            float(maximum_session_factor),
        )
        weights.append(class_factor * session_factor)
    result = torch.tensor(weights, dtype=torch.double)
    return result / result.mean()


def passage_direction_sampling_weights(
    labels: Sequence[int],
    class_names: Sequence[str],
    maximum_direction_factor: float = 2.0,
) -> torch.Tensor:
    if not labels:
        raise ValueError("labels must be non-empty")
    counts = passage_label_counts(labels, class_names)
    positive = torch.as_tensor(
        counts["direction_positive"],
        dtype=torch.float64,
    )
    available = positive > 0
    direction_factors = torch.ones(3, dtype=torch.float64)
    if available.any():
        largest = positive[available].max()
        direction_factors[available] = torch.sqrt(
            largest / positive[available]
        ).clamp(max=float(maximum_direction_factor))
    weights = []
    for label in labels:
        target = passage_target_from_index(label, class_names)
        positive_directions = target["directions"].bool()
        if positive_directions.any():
            weights.append(float(direction_factors[positive_directions].max()))
        else:
            weights.append(1.0)
    result = torch.as_tensor(weights, dtype=torch.double)
    return result / result.mean()


class PassageDirectionLoss(nn.Module):
    def __init__(
        self,
        direction_pos_weight: torch.Tensor = None,
        direction_loss_weight: float = 1.0,
    ):
        super().__init__()
        direction_weight = (
            torch.ones(3, dtype=torch.float32)
            if direction_pos_weight is None
            else torch.as_tensor(direction_pos_weight, dtype=torch.float32)
        )
        if direction_weight.shape != (3,):
            raise ValueError("direction_pos_weight must contain 3 values")
        self.register_buffer("direction_pos_weight", direction_weight)
        self.direction_loss_weight = float(direction_loss_weight)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict):
        direction_logits = outputs["direction_logits"]
        direction_targets = targets["directions"].to(
            dtype=direction_logits.dtype
        )
        direction_loss = nn.functional.binary_cross_entropy_with_logits(
            direction_logits,
            direction_targets,
            pos_weight=self.direction_pos_weight,
        )
        return self.direction_loss_weight * direction_loss


def last_blocks_for_epoch(
    epoch: int,
    freeze_backbone_epochs: int,
    schedule: Sequence[Dict],
) -> int:
    if int(epoch) <= int(freeze_backbone_epochs):
        return 0
    last_blocks = 0
    for entry in schedule:
        if int(epoch) >= int(entry["epoch"]):
            last_blocks = int(entry["last_blocks"])
    return last_blocks


def configure_trainable_layers(model, last_blocks: int) -> Dict[str, int]:
    dino = getattr(model, "dino", model)
    if not hasattr(dino, "head") or not hasattr(dino, "blocks"):
        raise ValueError("DINO model must expose head and blocks modules")
    block_count = len(dino.blocks)
    if last_blocks < 0 or last_blocks > block_count:
        raise ValueError(
            f"last_blocks must be in [0, {block_count}]: {last_blocks}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    task_parameters = (
        model.task_parameters()
        if hasattr(model, "task_parameters")
        else dino.head.parameters()
    )
    for parameter in task_parameters:
        parameter.requires_grad = True

    if last_blocks >= block_count:
        for parameter in dino.parameters():
            parameter.requires_grad = True
    elif last_blocks > 0:
        for block in dino.blocks[-last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        if hasattr(dino, "norm"):
            for parameter in dino.norm.parameters():
                parameter.requires_grad = True

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "last_blocks": int(last_blocks),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
    }


def parameter_groups(
    model,
    head_learning_rate: float,
    backbone_learning_rate: float,
):
    dino = getattr(model, "dino", model)
    task_parameters = list(
        model.task_parameters()
        if hasattr(model, "task_parameters")
        else dino.head.parameters()
    )
    task_ids = {id(parameter) for parameter in task_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in task_ids
    ]
    return [
        {
            "params": backbone_parameters,
            "lr": float(backbone_learning_rate),
            "name": "backbone",
        },
        {
            "params": task_parameters,
            "lr": float(head_learning_rate),
            "name": "head",
        },
    ]


def create_optimizer(model, config: Dict):
    groups = parameter_groups(
        model,
        config["head_learning_rate"],
        config["backbone_learning_rate"],
    )
    common = {
        "params": groups,
        "weight_decay": float(config["weight_decay"]),
    }
    name = str(config["name"]).lower()
    if name == "adamw":
        return AdamW(
            **common,
            betas=tuple(config["betas"]),
            eps=float(config["epsilon"]),
        )
    if name == "adam":
        return Adam(
            **common,
            betas=tuple(config["betas"]),
            eps=float(config["epsilon"]),
        )
    if name == "sgd":
        return SGD(
            **common,
            momentum=float(config["momentum"]),
        )
    raise ValueError(f"unsupported optimizer: {name}")


def learning_rate_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    warmup_start_factor: float,
    scheduler_name: str,
    min_learning_rate_ratio: float,
) -> float:
    step = max(0, int(step))
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")

    if warmup_steps > 0 and step < warmup_steps:
        progress = step / max(1, warmup_steps - 1)
        return float(
            warmup_start_factor
            + (1.0 - warmup_start_factor) * progress
        )

    name = str(scheduler_name).lower()
    if name == "constant":
        return 1.0
    if name != "cosine":
        raise ValueError(f"unsupported scheduler: {name}")

    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / max(1, decay_steps - 1)
    progress = min(1.0, max(0.0, progress))
    cosine_factor = 0.5 * (1.0 + cos(pi * progress))
    return float(
        min_learning_rate_ratio
        + (1.0 - min_learning_rate_ratio) * cosine_factor
    )


def create_scheduler(
    optimizer,
    config: Dict,
    total_epochs: int,
    steps_per_epoch: int,
):
    steps_per_epoch = int(steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    total_steps = int(total_epochs) * steps_per_epoch
    warmup_steps = int(config["warmup_epochs"]) * steps_per_epoch

    def multiplier(step):
        return learning_rate_multiplier(
            step=step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_start_factor=float(config["warmup_start_factor"]),
            scheduler_name=config["name"],
            min_learning_rate_ratio=float(
                config["min_learning_rate_ratio"]
            ),
        )

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _macro_f1(confusion: torch.Tensor) -> float:
    scores = []
    for index in range(confusion.shape[0]):
        true_positive = float(confusion[index, index])
        false_positive = float(confusion[:, index].sum() - confusion[index, index])
        false_negative = float(confusion[index, :].sum() - confusion[index, index])
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0.0 else 2.0 * true_positive / denominator
        )
    return float(sum(scores) / len(scores))


def run_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    optimizer=None,
    scheduler=None,
    scaler=None,
    use_amp: bool = False,
    description: str = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    correct = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    progress = tqdm(
        loader,
        desc=description or ("train" if training else "test"),
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for inputs, labels in progress:
        if isinstance(inputs, dict):
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in inputs.items()
            }
        else:
            inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast_context:
            logits = model(inputs)
            loss = criterion(logits, labels)

        if training:
            optimizer_stepped = True
            if scaler is not None and scaler.is_enabled():
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= previous_scale
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None and optimizer_stepped:
                scheduler.step()

        predictions = torch.argmax(logits.detach(), dim=1)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        correct += int((predictions == labels).sum())
        for target, prediction in zip(
            labels.detach().cpu().tolist(),
            predictions.cpu().tolist(),
        ):
            confusion[int(target), int(prediction)] += 1
        postfix = {
            "loss": f"{total_loss / total_samples:.4f}",
            "acc": f"{correct / total_samples:.3f}",
        }
        if training:
            head_group = next(
                (
                    group
                    for group in optimizer.param_groups
                    if group.get("name") == "head"
                ),
                optimizer.param_groups[0],
            )
            postfix["lr"] = f"{float(head_group['lr']):.2e}"
        progress.set_postfix(postfix)

    if total_samples == 0:
        raise ValueError("data loader contains no samples")
    return {
        "loss": total_loss / total_samples,
        "accuracy": correct / total_samples,
        "macro_f1": _macro_f1(confusion),
    }


def _binary_f1(true_positive: float, false_positive: float, false_negative: float):
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0.0 else 2.0 * true_positive / denominator


def run_passage_epoch(
    model,
    loader,
    criterion,
    device,
    direction_thresholds: Sequence[float],
    optimizer=None,
    scheduler=None,
    scaler=None,
    use_amp: bool = False,
    description: str = None,
) -> Dict[str, float]:
    thresholds = torch.as_tensor(direction_thresholds, dtype=torch.float32)
    if thresholds.shape != (3,):
        raise ValueError("direction_thresholds must contain 3 values")
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    direction_tp = torch.zeros(3, dtype=torch.int64)
    direction_fp = torch.zeros(3, dtype=torch.int64)
    direction_fn = torch.zeros(3, dtype=torch.int64)
    direction_exact_correct = 0

    progress = tqdm(
        loader,
        desc=description or ("train" if training else "test"),
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for inputs, targets in progress:
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in inputs.items()
        }
        targets = {
            key: value.to(device, non_blocking=True)
            for key, value in targets.items()
        }
        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast_context:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if training:
            optimizer_stepped = True
            if scaler is not None and scaler.is_enabled():
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= previous_scale
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None and optimizer_stepped:
                scheduler.step()

        direction_targets = targets["directions"].detach().cpu().bool()
        batch_size = int(direction_targets.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        direction_predictions = (
            torch.sigmoid(outputs["direction_logits"].detach()).cpu()
            >= thresholds
        )
        direction_tp += (direction_predictions & direction_targets).sum(dim=0)
        direction_fp += (direction_predictions & ~direction_targets).sum(dim=0)
        direction_fn += (~direction_predictions & direction_targets).sum(dim=0)
        direction_exact_correct += int(
            (direction_predictions == direction_targets).all(dim=1).sum()
        )
        progress.set_postfix(
            {
                "loss": f"{total_loss / total_samples:.4f}",
                "dir_exact": f"{direction_exact_correct / total_samples:.3f}",
            }
        )

    if total_samples == 0:
        raise ValueError("data loader contains no samples")
    direction_f1 = [
        _binary_f1(
            float(direction_tp[index]),
            float(direction_fp[index]),
            float(direction_fn[index]),
        )
        for index in range(3)
    ]
    direction_macro_f1 = sum(direction_f1) / len(direction_f1)
    return {
        "loss": total_loss / total_samples,
        "direction_front_f1": direction_f1[0],
        "direction_left_f1": direction_f1[1],
        "direction_right_f1": direction_f1[2],
        "direction_macro_f1": direction_macro_f1,
        "direction_exact_accuracy": direction_exact_correct / total_samples,
        "passage_macro_f1": direction_macro_f1,
    }


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    model_config: Dict,
    training_config: Dict,
    optimizer_config: Dict,
    scheduler_config: Dict,
    metrics: Dict,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "class_names": list(model_config["class_names"]),
            "model_name": str(model_config["model_name"]),
            "input_size": list(model_config["input_size"]),
            "num_classes": int(model_config["num_classes"]),
            "output_mode": str(model_config.get("output_mode", "class")),
            "architecture": str(model_config.get("architecture", "rgb")),
            "variant": {
                key: model_config[key]
                for key in (
                    "sequence_length",
                    "dino_readout",
                    "frame_stride",
                    "maximum_gap_seconds",
                    "use_depth",
                    "use_gru",
                    "depth_feature_dim",
                    "depth_pool_size",
                    "depth_min_m",
                    "depth_max_m",
                    "fusion_dim",
                    "gru_hidden_size",
                    "gru_num_layers",
                    "direction_names",
                    "direction_thresholds",
                    "turning_class_name",
                )
                if key in model_config
            },
            "metrics": dict(metrics),
            "training": dict(training_config),
            "optimizer": dict(optimizer_config),
            "scheduler": dict(scheduler_config),
        },
        path,
    )


def write_effective_config(path: str, config: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


class MetricsWriter:
    CLASSIFICATION_FIELDS = (
        "epoch",
        "last_blocks",
        "trainable_parameters",
        "backbone_learning_rate",
        "head_learning_rate",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "test_loss",
        "test_accuracy",
        "test_macro_f1",
    )

    PASSAGE_FIELDS = (
        "epoch",
        "last_blocks",
        "trainable_parameters",
        "backbone_learning_rate",
        "head_learning_rate",
        "train_loss",
        "train_direction_front_f1",
        "train_direction_left_f1",
        "train_direction_right_f1",
        "train_direction_macro_f1",
        "train_direction_exact_accuracy",
        "train_passage_macro_f1",
        "test_loss",
        "test_direction_front_f1",
        "test_direction_left_f1",
        "test_direction_right_f1",
        "test_direction_macro_f1",
        "test_direction_exact_accuracy",
        "test_passage_macro_f1",
    )

    def __init__(self, path: str, fields: Sequence[str] = None):
        self.path = os.path.abspath(path)
        self.fields = tuple(fields or self.CLASSIFICATION_FIELDS)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._stream = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=self.fields)
        self._writer.writeheader()
        self._stream.flush()

    def write(self, values: Dict) -> None:
        self._writer.writerow({key: values.get(key) for key in self.fields})
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()
