import os
from typing import Any, Dict

import yaml

from corridor_classifier.passage_directions import (
    CLASS_TO_DIRECTIONS,
    DIRECTION_NAMES,
)


ARCHITECTURES = ("rgb", "rgb_gru", "rgb_depth", "rgb_depth_gru")


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

    architecture = str(model.get("architecture", "rgb")).strip()
    if architecture not in ARCHITECTURES:
        raise ValueError(
            "model.architecture must be one of " + ", ".join(ARCHITECTURES)
        )
    variants = model.get("variants", {})
    if variants:
        if not isinstance(variants, dict) or architecture not in variants:
            raise ValueError(
                f"model.variants must define the active {architecture} variant"
            )
        model.update(dict(variants[architecture]))
    model["architecture"] = architecture

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
    output_mode = str(model.get("output_mode", "class")).strip().lower()
    if output_mode not in ("class", "passage_directions"):
        raise ValueError(
            "model.output_mode must be class or passage_directions"
        )
    model["output_mode"] = output_mode
    dino_readout = str(model.get("dino_readout", "last_cls")).strip().lower()
    valid_dino_readouts = (
        "last_cls",
        "last_cls_patch_mean",
        "last4_cls",
        "last4_cls_patch_mean",
    )
    if dino_readout not in valid_dino_readouts:
        raise ValueError(
            "model.dino_readout must be one of "
            + ", ".join(valid_dino_readouts)
        )
    model["dino_readout"] = dino_readout
    if output_mode == "passage_directions":
        missing_passage_classes = [
            name for name in CLASS_TO_DIRECTIONS if name not in class_names
        ]
        if missing_passage_classes:
            raise ValueError(
                "passage_directions requires source class(es): "
                + ", ".join(missing_passage_classes)
            )
        direction_names = model.get("direction_names", list(DIRECTION_NAMES))
        if list(direction_names) != list(DIRECTION_NAMES):
            raise ValueError(
                "model.direction_names must be [front, left, right]"
            )
        direction_thresholds = [
            float(value)
            for value in model.get("direction_thresholds", [0.5, 0.5, 0.5])
        ]
        if len(direction_thresholds) != 3 or any(
            not 0.0 < value < 1.0 for value in direction_thresholds
        ):
            raise ValueError(
                "model.direction_thresholds must contain 3 values in (0, 1)"
            )
        turning_class_name = str(
            model.get("turning_class_name", "turning")
        ).strip()
        if turning_class_name not in class_names:
            raise ValueError(
                "model.turning_class_name must exist in model.class_names"
            )
        turning_threshold = float(model.get("turning_threshold", 0.5))
        if not 0.0 < turning_threshold < 1.0:
            raise ValueError("model.turning_threshold must be in (0, 1)")
        model.update(
            {
                "direction_names": list(DIRECTION_NAMES),
                "direction_thresholds": direction_thresholds,
                "turning_class_name": turning_class_name,
                "turning_threshold": turning_threshold,
            }
        )
    model["sequence_length"] = int(model.get("sequence_length", 1))
    if model["sequence_length"] <= 0:
        raise ValueError("model.sequence_length must be positive")
    model["frame_stride"] = int(model.get("frame_stride", 1))
    if model["frame_stride"] <= 0:
        raise ValueError("model.frame_stride must be positive")
    model["use_depth"] = bool(model.get("use_depth", False))
    model["use_gru"] = bool(model.get("use_gru", False))
    if model["use_gru"] != architecture.endswith("gru"):
        raise ValueError("model use_gru does not match architecture")
    if model["use_depth"] != ("depth" in architecture):
        raise ValueError("model use_depth does not match architecture")
    if not model["use_gru"] and model["sequence_length"] != 1:
        raise ValueError("non-GRU architectures require sequence_length=1")
    if not model["use_gru"] and model["frame_stride"] != 1:
        raise ValueError("non-GRU architectures require frame_stride=1")
    model["maximum_gap_seconds"] = float(
        model.get("maximum_gap_seconds", 0.4)
    )
    if model["maximum_gap_seconds"] <= 0.0:
        raise ValueError("model.maximum_gap_seconds must be positive")
    if model["use_depth"]:
        model["depth_min_m"] = float(model.get("depth_min_m", 0.1))
        model["depth_max_m"] = float(model.get("depth_max_m", 10.0))
        model["depth_feature_dim"] = int(
            model.get("depth_feature_dim", 128)
        )
        model["depth_pool_size"] = int(model.get("depth_pool_size", 1))
        if not 0.0 < model["depth_min_m"] < model["depth_max_m"]:
            raise ValueError("depth range must satisfy 0 < min < max")
        if model["depth_feature_dim"] <= 0:
            raise ValueError("model.depth_feature_dim must be positive")
        if model["depth_pool_size"] <= 0:
            raise ValueError("model.depth_pool_size must be positive")
    if architecture != "rgb":
        model["fusion_dim"] = int(model.get("fusion_dim", 256))
        if model["fusion_dim"] <= 0:
            raise ValueError("model.fusion_dim must be positive")
    if model["use_gru"]:
        model["gru_hidden_size"] = int(model.get("gru_hidden_size", 256))
        model["gru_num_layers"] = int(model.get("gru_num_layers", 1))
        if model["gru_hidden_size"] <= 0 or model["gru_num_layers"] <= 0:
            raise ValueError("GRU dimensions must be positive")

    inference_rate = float(runtime.get("inference_rate", 0.0))
    if inference_rate <= 0.0:
        raise ValueError("runtime.inference_rate must be positive")
    runtime["inference_rate"] = inference_rate

    required_topic_keys = (
        "image_topic",
        "passage_type_topic",
        "probabilities_topic",
    )
    if model["use_depth"]:
        required_topic_keys += ("depth_topic",)
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
    depth_generation = dict(dataset_data.get("depth_generation", {}))
    turn_detection = dict(dataset_data.get("turn_detection", {}))
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
    dataset_types = depth_generation.get("dataset_types", ["train"])
    if not isinstance(dataset_types, list) or not dataset_types:
        raise ValueError("depth_generation.dataset_types must be a non-empty list")
    dataset_types = [str(value) for value in dataset_types]
    if any(value not in ("train", "test") for value in dataset_types):
        raise ValueError("depth_generation.dataset_types must contain train/test")
    session_names = depth_generation.get("session_names", [])
    if not isinstance(session_names, list):
        raise ValueError("depth_generation.session_names must be a list")
    unidepth_config_file = str(
        depth_generation.get("unidepth_config_file", "")
    ).strip()
    if not unidepth_config_file:
        raise ValueError("depth_generation.unidepth_config_file must not be empty")
    stamp_tolerance = float(
        depth_generation.get("stamp_tolerance_seconds", 0.01)
    )
    if stamp_tolerance < 0.0:
        raise ValueError(
            "depth_generation.stamp_tolerance_seconds must be non-negative"
        )
    depth_generation.update(
        {
            "dataset_types": dataset_types,
            "session_names": [str(value) for value in session_names],
            "unidepth_config_file": unidepth_config_file,
            "stamp_tolerance_seconds": stamp_tolerance,
        }
    )
    turn_enabled = bool(turn_detection.get("enabled", False))
    source_num_classes = int(
        turn_detection.get("source_num_classes", config["model"]["num_classes"])
    )
    turn_class_name = str(
        turn_detection.get("class_name", "turning")
    ).strip()
    pose_topic = str(turn_detection.get("pose_topic", "/mcl_pose")).strip()
    if source_num_classes <= 0 or source_num_classes > config["model"]["num_classes"]:
        raise ValueError("turn_detection.source_num_classes is invalid")
    if turn_enabled and turn_class_name not in config["model"]["class_names"]:
        raise ValueError("turn_detection.class_name must exist in model.class_names")
    if turn_enabled and not pose_topic:
        raise ValueError("turn_detection.pose_topic must not be empty")
    threshold = float(
        turn_detection.get("angular_speed_threshold_rad_s", 0.20)
    )
    window_seconds = float(turn_detection.get("window_seconds", 1.0))
    minimum_duration = float(
        turn_detection.get("minimum_duration_seconds", 0.5)
    )
    padding_seconds = float(turn_detection.get("padding_seconds", 0.25))
    maximum_pose_gap = float(
        turn_detection.get("maximum_pose_gap_seconds", 0.5)
    )
    post_turn_maximum = float(
        turn_detection.get("post_turn_next_label_max_seconds", 6.0)
    )
    turning_gap_maximum = float(
        turn_detection.get("turning_gap_bridge_max_seconds", 1.5)
    )
    if threshold <= 0.0 or window_seconds <= 0.0 or maximum_pose_gap <= 0.0:
        raise ValueError(
            "turn detection threshold, window, and pose gap must be positive"
        )
    if minimum_duration < 0.0 or padding_seconds < 0.0:
        raise ValueError("turn detection duration and padding must be non-negative")
    if post_turn_maximum < 0.0:
        raise ValueError(
            "turn_detection.post_turn_next_label_max_seconds must be non-negative"
        )
    if turning_gap_maximum < 0.0:
        raise ValueError(
            "turn_detection.turning_gap_bridge_max_seconds must be non-negative"
        )
    turn_detection.update(
        {
            "enabled": turn_enabled,
            "source_num_classes": source_num_classes,
            "class_name": turn_class_name,
            "class_index": (
                config["model"]["class_names"].index(turn_class_name)
                if turn_enabled
                else None
            ),
            "pose_topic": pose_topic,
            "angular_speed_threshold_rad_s": threshold,
            "window_seconds": window_seconds,
            "minimum_duration_seconds": minimum_duration,
            "padding_seconds": padding_seconds,
            "maximum_pose_gap_seconds": maximum_pose_gap,
            "post_turn_next_label_max_seconds": post_turn_maximum,
            "turning_gap_bridge_max_seconds": turning_gap_maximum,
        }
    )
    config["paths"] = paths
    config["collection"] = collection
    config["depth_generation"] = depth_generation
    config["turn_detection"] = turn_detection
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
    for key in ("train_session_names", "test_session_names"):
        session_names = dataset.get(key, [])
        if session_names is None:
            session_names = []
        if not isinstance(session_names, list):
            raise ValueError(f"dataset.{key} must be a list")
        dataset[key] = [str(name).strip() for name in session_names]
        if any(not name for name in dataset[key]):
            raise ValueError(f"dataset.{key} must not contain empty names")
    dataset["num_workers"] = int(dataset.get("num_workers", 0))
    if dataset["num_workers"] < 0:
        raise ValueError("dataset.num_workers must be non-negative")

    required_training_keys = (
        "epochs",
        "use_amp",
        "seed",
        "freeze_backbone_epochs",
        "unfreeze_schedule",
        "pretrained_weights_path",
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
    batch_sizes = training.get("batch_size_by_architecture", {})
    if batch_sizes:
        if config["model"]["architecture"] not in batch_sizes:
            raise ValueError(
                "training.batch_size_by_architecture is missing active architecture"
            )
        training["batch_size"] = int(
            batch_sizes[config["model"]["architecture"]]
        )
    elif "batch_size" not in training:
        raise ValueError("training must define batch_size or batch_size_by_architecture")
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
    checkpoint_path = str(config["model"]["checkpoint_path"])
    checkpoint_stem, checkpoint_extension = os.path.splitext(checkpoint_path)
    training["output_checkpoint"] = checkpoint_path
    training["final_checkpoint"] = (
        checkpoint_stem + "_final" + (checkpoint_extension or ".pth")
    )
    run_root = str(
        training.get("run_root", "runs/corridor_classifier")
    ).strip()
    if not run_root:
        raise ValueError("training.run_root must not be empty")
    training["metrics_path"] = os.path.join(
        run_root,
        config["model"]["architecture"],
        "metrics.csv",
    )
    config["dataset"] = dataset
    config["training"] = training
    config["optimizer"] = optimizer
    config["scheduler"] = scheduler
    return config
