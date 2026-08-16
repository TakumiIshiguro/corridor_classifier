import os
from copy import deepcopy

import pytest

from corridor_classifier.config import (
    load_collection_config,
    load_config,
    load_training_config,
    package_root,
    resolve_path,
    _validate_config,
    load_yaml,
)


def test_default_config_has_nine_unique_classes_including_turning():
    config = load_config(os.path.join(package_root(), "config"))

    assert config["model"]["model_name"] == "vit_small_patch14_dinov2.lvd142m"
    assert config["model"]["architecture"] == "rgb_gru"
    assert config["model"]["use_depth"] is False
    assert config["model"]["use_gru"] is True
    assert config["model"]["frame_stride"] == 1
    assert config["model"]["input_size"] == [224, 224]
    assert config["model"]["num_classes"] == 9
    assert len(set(config["model"]["class_names"])) == 9
    assert config["model"]["class_names"][-1] == "turning"
    assert config["runtime"] == {"inference_rate": 4.0}
    assert config["topics"]["image_topic"] == "/camera_center/image_raw"
    assert config["topics"]["label_topic"] == "/cmd_dir_intersection"
    assert config["topics"]["passage_type_topic"] == "/passage_type"


def test_resolve_path_uses_package_root_for_relative_path():
    expected = os.path.join(package_root(), "weights", "corridor_classifier.pth")
    assert resolve_path("weights/corridor_classifier.pth") == expected


def test_class_count_mismatch_is_rejected(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "topics.yaml").write_text(
        "image_topic: /image\n"
        "label_topic: /labels\n"
        "passage_type_topic: /passage\n"
        "probabilities_topic: /probabilities\n"
    )
    (config_dir / "model.yaml").write_text(
        "model:\n"
        "  model_name: test\n"
        "  checkpoint_path: model.pth\n"
        "  input_size: [224, 224]\n"
        "  num_classes: 8\n"
        "  class_names: [a, b]\n"
        "  device: cpu\n"
        "  use_fp16: false\n"
        "  strict_checkpoint: true\n"
        "runtime:\n"
        "  inference_rate: 1.0\n"
    )

    with pytest.raises(ValueError, match="class_names length"):
        load_config(str(config_dir))


def test_collection_and_training_configs_match_model_input():
    config_dir = os.path.join(package_root(), "config")
    collection = load_collection_config(config_dir)
    training = load_training_config(config_dir)

    assert collection["model"]["input_size"] == [224, 224]
    assert collection["collection"]["source"] == "bag"
    assert collection["collection"]["bag_path"].endswith(".bag")
    assert collection["collection"]["sample_dt"] == 0.25
    assert collection["turn_detection"]["enabled"] is True
    assert collection["turn_detection"]["source_num_classes"] == 8
    assert collection["turn_detection"]["class_index"] == 8
    assert (
        collection["turn_detection"]["post_turn_next_label_max_seconds"]
        == 6.0
    )
    assert collection["turn_detection"]["turning_gap_bridge_max_seconds"] == 1.5
    assert training["dataset"]["train_data_dir"]
    assert training["dataset"]["train_session_names"] == [
        "session_20260811_010115"
    ]
    assert isinstance(training["training"]["use_test"], bool)
    assert (
        0
        <= training["training"]["freeze_backbone_epochs"]
        <= training["training"]["epochs"]
    )
    assert isinstance(training["training"]["unfreeze_schedule"], list)
    assert training["optimizer"]["name"] in ("adamw", "adam", "sgd")
    assert training["optimizer"]["head_learning_rate"] > 0.0
    assert training["optimizer"]["backbone_learning_rate"] > 0.0
    assert training["scheduler"]["name"] in ("cosine", "constant")
    assert (
        0
        <= training["scheduler"]["warmup_epochs"]
        < training["training"]["epochs"]
    )


@pytest.mark.parametrize(
    "architecture,use_depth,use_gru,sequence_length",
    [
        ("rgb", False, False, 1),
        ("rgb_gru", False, True, 5),
        ("rgb_depth", True, False, 1),
        ("rgb_depth_gru", True, True, 5),
    ],
)
def test_all_architecture_configs_are_valid(
    architecture, use_depth, use_gru, sequence_length
):
    config_dir = os.path.join(package_root(), "config")
    model_data = load_yaml(os.path.join(config_dir, "model.yaml"))
    model = deepcopy(model_data["model"])
    model["architecture"] = architecture
    config = {
        "model": model,
        "runtime": deepcopy(model_data["runtime"]),
        "topics": load_yaml(os.path.join(config_dir, "topics.yaml")),
    }

    _validate_config(config)

    assert config["model"]["use_depth"] is use_depth
    assert config["model"]["use_gru"] is use_gru
    assert config["model"]["sequence_length"] == sequence_length
    assert config["model"]["frame_stride"] == 1
