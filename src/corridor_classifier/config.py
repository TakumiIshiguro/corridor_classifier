import os
from typing import Any, Dict

import yaml


def package_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def resolve_path(path: str, base_dir: str = None) -> str:
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = package_root()
    return os.path.abspath(os.path.join(base_dir, path))


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _validate_config(config: Dict[str, Any]) -> None:
    model = config["model"]
    runtime = config["runtime"]
    topics = config["topics"]

    required_model_keys = (
        "model_name",
        "checkpoint_path",
        "input_size",
        "num_classes",
        "class_names",
        "device",
        "use_fp16",
        "strict_checkpoint",
    )
    missing = [key for key in required_model_keys if key not in model]
    if missing:
        raise ValueError(f"model config is missing keys: {', '.join(missing)}")

    input_size = model["input_size"]
    if (
        not isinstance(input_size, list)
        or len(input_size) != 2
        or any(int(value) <= 0 for value in input_size)
    ):
        raise ValueError("model.input_size must contain two positive integers")
    model["input_size"] = [int(input_size[0]), int(input_size[1])]

    num_classes = int(model["num_classes"])
    class_names = model["class_names"]
    if not isinstance(class_names, list) or len(class_names) != num_classes:
        raise ValueError(
            "model.class_names length must equal model.num_classes "
            f"({len(class_names) if isinstance(class_names, list) else 'invalid'} "
            f"!= {num_classes})"
        )
    if len(set(str(name) for name in class_names)) != num_classes:
        raise ValueError("model.class_names must contain unique names")
    model["num_classes"] = num_classes
    model["class_names"] = [str(name) for name in class_names]

    inference_rate = float(runtime.get("inference_rate", 0.0))
    if inference_rate <= 0.0:
        raise ValueError("runtime.inference_rate must be positive")
    runtime["inference_rate"] = inference_rate

    required_topic_keys = (
        "image_topic",
        "passage_type_topic",
        "probabilities_topic",
    )
    missing = [key for key in required_topic_keys if not topics.get(key)]
    if missing:
        raise ValueError(f"topics config is missing keys: {', '.join(missing)}")


def load_config(config_dir: str = None) -> Dict[str, Any]:
    if config_dir is None:
        config_dir = os.path.join(package_root(), "config")
    config_dir = os.path.abspath(os.path.expanduser(config_dir))

    model_data = load_yaml(os.path.join(config_dir, "model.yaml"))
    topics = load_yaml(os.path.join(config_dir, "topics.yaml"))
    if "model" not in model_data:
        raise ValueError("model.yaml must contain a 'model' mapping")

    config = {
        "model": dict(model_data["model"]),
        "runtime": dict(model_data.get("runtime", {})),
        "topics": topics,
    }
    _validate_config(config)
    return config


def load_collection_config(config_dir: str = None) -> Dict[str, Any]:
    config = load_config(config_dir)
    if config_dir is None:
        config_dir = os.path.join(package_root(), "config")
    dataset_data = load_yaml(
        os.path.join(
            os.path.abspath(os.path.expanduser(config_dir)),
            "dataset.yaml",
        )
    )
    paths = dict(dataset_data.get("paths", {}))
    collection = dict(dataset_data.get("collection", {}))
    if not paths.get("dataset_dir"):
        raise ValueError("dataset.yaml must define paths.dataset_dir")
    if not config["topics"].get("label_topic"):
        raise ValueError("topics.yaml must define label_topic")

    source = str(collection.get("source", "live")).strip().lower()
    if source not in ("live", "bag"):
        raise ValueError("collection.source must be live or bag")
    bag_path = str(collection.get("bag_path", "")).strip()
    if source == "bag" and not bag_path:
        raise ValueError(
            "collection.bag_path must be set when collection.source is bag"
        )
    dataset_type = str(collection.get("dataset_type", "")).strip()
    if dataset_type not in ("train", "test"):
        raise ValueError(
            "collection.dataset_type must be train or test"
        )
    sample_dt = float(collection.get("sample_dt", 0.0))
    label_timeout = float(collection.get("label_timeout", 0.0))
    if sample_dt <= 0.0:
        raise ValueError("collection.sample_dt must be positive")
    if label_timeout <= 0.0:
        raise ValueError("collection.label_timeout must be positive")
    image_format = str(collection.get("image_format", "jpg")).lower()
    if image_format not in ("jpg", "jpeg", "png"):
        raise ValueError("collection.image_format must be jpg, jpeg, or png")
    jpeg_quality = int(collection.get("jpeg_quality", 95))
    if jpeg_quality < 1 or jpeg_quality > 100:
        raise ValueError("collection.jpeg_quality must be in [1, 100]")

    collection.update(
        {
            "source": source,
            "bag_path": bag_path,
            "dataset_type": dataset_type,
            "session_name": str(collection.get("session_name", "")).strip(),
            "sample_dt": sample_dt,
            "label_timeout": label_timeout,
            "image_format": image_format,
            "jpeg_quality": jpeg_quality,
        }
    )
    config["paths"] = paths
    config["collection"] = collection
    return config


def load_training_config(config_dir: str = None) -> Dict[str, Any]:
    config = load_config(config_dir)
    if config_dir is None:
        config_dir = os.path.join(package_root(), "config")
    training_data = load_yaml(
        os.path.join(
            os.path.abspath(os.path.expanduser(config_dir)),
            "training.yaml",
        )
    )
    dataset = dict(training_data.get("dataset", {}))
    training = dict(training_data.get("training", {}))
    optimizer = dict(training_data.get("optimizer", {}))
    scheduler = dict(training_data.get("scheduler", {}))
    train_data_dir = str(dataset.get("train_data_dir", "")).strip()
    test_data_dir = str(dataset.get("test_data_dir", "") or "").strip()
    if not train_data_dir:
        raise ValueError("training.yaml must define dataset.train_data_dir")
    dataset["train_data_dir"] = train_data_dir
    dataset["test_data_dir"] = test_data_dir
    dataset["num_workers"] = int(dataset.get("num_workers", 0))
    if dataset["num_workers"] < 0:
        raise ValueError("dataset.num_workers must be non-negative")

    required_training_keys = (
        "epochs",
        "batch_size",
        "use_amp",
        "seed",
        "freeze_backbone_epochs",
        "unfreeze_schedule",
        "pretrained_weights_path",
        "output_checkpoint",
        "final_checkpoint",
        "metrics_path",
    )
    missing = [key for key in required_training_keys if key not in training]
    if missing:
        raise ValueError(
            f"training config is missing keys: {', '.join(missing)}"
        )

    epochs = int(training["epochs"])
    freeze_epochs = int(training["freeze_backbone_epochs"])
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if freeze_epochs < 0 or freeze_epochs > epochs:
        raise ValueError(
            "training.freeze_backbone_epochs must be in [0, epochs]"
        )
    if int(training["batch_size"]) <= 0:
        raise ValueError("training.batch_size must be positive")
    use_test = bool(training.get("use_test", False))
    if use_test and not test_data_dir:
        raise ValueError(
            "dataset.test_data_dir is required when training.use_test is true"
        )

    schedule = training["unfreeze_schedule"]
    if schedule is None:
        schedule = []
    if not isinstance(schedule, list):
        raise ValueError("training.unfreeze_schedule must be a list")
    previous_epoch = freeze_epochs
    normalized_schedule = []
    for entry in schedule:
        if not isinstance(entry, dict):
            raise ValueError("each unfreeze schedule entry must be a mapping")
        epoch = int(entry.get("epoch", 0))
        last_blocks = int(entry.get("last_blocks", 0))
        if epoch <= previous_epoch or epoch > epochs:
            raise ValueError(
                "unfreeze schedule epochs must be strictly increasing, "
                "after freeze_backbone_epochs, and no greater than epochs"
            )
        if last_blocks <= 0:
            raise ValueError("unfreeze schedule last_blocks must be positive")
        normalized_schedule.append(
            {"epoch": epoch, "last_blocks": last_blocks}
        )
        previous_epoch = epoch

    optimizer_name = str(optimizer.get("name", "")).strip().lower()
    if optimizer_name not in ("adamw", "adam", "sgd"):
        raise ValueError("optimizer.name must be adamw, adam, or sgd")
    for key in ("head_learning_rate", "backbone_learning_rate"):
        value = float(optimizer.get(key, 0.0))
        if value <= 0.0:
            raise ValueError(f"optimizer.{key} must be positive")
        optimizer[key] = value
    weight_decay = float(optimizer.get("weight_decay", 0.0))
    if weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative")
    optimizer["name"] = optimizer_name
    optimizer["weight_decay"] = weight_decay
    if optimizer_name in ("adamw", "adam"):
        betas = optimizer.get("betas", [])
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or any(not 0.0 <= float(value) < 1.0 for value in betas)
        ):
            raise ValueError(
                "optimizer.betas must contain two values in [0, 1)"
            )
        epsilon = float(optimizer.get("epsilon", 0.0))
        if epsilon <= 0.0:
            raise ValueError("optimizer.epsilon must be positive")
        optimizer["betas"] = [float(betas[0]), float(betas[1])]
        optimizer["epsilon"] = epsilon
    else:
        momentum = float(optimizer.get("momentum", 0.0))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("optimizer.momentum must be in [0, 1)")
        optimizer["momentum"] = momentum

    scheduler_name = str(scheduler.get("name", "")).strip().lower()
    if scheduler_name not in ("cosine", "constant"):
        raise ValueError("scheduler.name must be cosine or constant")
    warmup_epochs = int(scheduler.get("warmup_epochs", 0))
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError(
            "scheduler.warmup_epochs must be in [0, training.epochs)"
        )
    warmup_start_factor = float(
        scheduler.get("warmup_start_factor", 0.0)
    )
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError(
            "scheduler.warmup_start_factor must be in (0, 1]"
        )
    min_learning_rate_ratio = float(
        scheduler.get("min_learning_rate_ratio", 0.0)
    )
    if not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError(
            "scheduler.min_learning_rate_ratio must be in [0, 1]"
        )
    scheduler.update(
        {
            "name": scheduler_name,
            "warmup_epochs": warmup_epochs,
            "warmup_start_factor": warmup_start_factor,
            "min_learning_rate_ratio": min_learning_rate_ratio,
        }
    )

    training.update(
        {
            "epochs": epochs,
            "batch_size": int(training["batch_size"]),
            "use_test": use_test,
            "use_amp": bool(training["use_amp"]),
            "seed": int(training["seed"]),
            "freeze_backbone_epochs": freeze_epochs,
            "unfreeze_schedule": normalized_schedule,
        }
    )
    config["dataset"] = dataset
    config["training"] = training
    config["optimizer"] = optimizer
    config["scheduler"] = scheduler
    return config
