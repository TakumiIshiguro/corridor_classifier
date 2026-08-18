import numpy as np
import pytest
import torch
from PIL import Image

from corridor_classifier.collection import DatasetSessionWriter
from corridor_classifier.dataset import (
    CorridorDataset,
    CorridorMultiInputDataset,
    CorridorSample,
    load_dataset_samples,
)


CLASS_NAMES = tuple(f"class_{index}" for index in range(8))


def make_session(root, split, name, class_index):
    writer = DatasetSessionWriter(
        session_dir=str(root / split / name),
        class_names=CLASS_NAMES,
        input_size=[224, 224],
        image_format="png",
        jpeg_quality=95,
        metadata={"dataset_type": split},
    )
    writer.save(
        Image.new("RGB", (320, 240)),
        class_index=class_index,
        stamp=1.0,
    )
    writer.close()


def test_train_and_test_datasets_are_loaded_separately(tmp_path):
    make_session(tmp_path, "train", "train_session", 3)
    make_session(tmp_path, "test", "test_session", 4)

    train_samples = load_dataset_samples(str(tmp_path / "train"), 8)
    test_samples = load_dataset_samples(str(tmp_path / "test"), 8)

    assert len(train_samples) == 1
    assert train_samples[0].class_index == 3
    assert len(test_samples) == 1
    assert test_samples[0].class_index == 4


def test_dataset_can_select_named_sessions(tmp_path):
    make_session(tmp_path, "train", "old_session", 1)
    make_session(tmp_path, "train", "latest_session", 6)

    samples = load_dataset_samples(
        str(tmp_path / "train"),
        8,
        session_names=["latest_session"],
    )

    assert len(samples) == 1
    assert samples[0].session_name == "latest_session"
    assert samples[0].class_index == 6


def test_missing_selected_session_is_rejected(tmp_path):
    make_session(tmp_path, "train", "existing_session", 1)

    with pytest.raises(ValueError, match="missing_session"):
        load_dataset_samples(
            str(tmp_path / "train"),
            8,
            session_names=["missing_session"],
        )


def test_dataset_returns_normalized_224_tensor(tmp_path):
    make_session(tmp_path, "train", "session_0", 3)
    train_samples = load_dataset_samples(str(tmp_path / "train"), 8)
    dataset = CorridorDataset(train_samples, [224, 224])

    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert label == 3


def test_temporal_depth_dataset_uses_final_label_and_stays_in_session(tmp_path):
    samples = []
    for index in range(3):
        image_path = tmp_path / f"{index}.png"
        depth_path = tmp_path / f"{index}.npy"
        Image.new("RGB", (224, 224)).save(image_path)
        np.save(depth_path, np.full((224, 224), index + 1, np.float32))
        samples.append(
            CorridorSample(
                image_path=str(image_path),
                depth_path=str(depth_path),
                class_index=index,
                session_name="session_a",
                stamp=1.0 + index * 0.25,
            )
        )
    dataset = CorridorMultiInputDataset(
        samples,
        [224, 224],
        {
            "sequence_length": 3,
            "maximum_gap_seconds": 0.4,
            "use_depth": True,
            "depth_min_m": 0.1,
            "depth_max_m": 10.0,
        },
    )

    inputs, label = dataset[0]
    assert inputs["rgb"].shape == (3, 3, 224, 224)
    assert inputs["depth"].shape == (3, 2, 224, 224)
    assert label == 2


def test_depth_dataset_rejects_samples_without_depth(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (224, 224)).save(image_path)
    sample = CorridorSample(str(image_path), 0, "session", stamp=1.0)

    with pytest.raises(ValueError, match="no valid sequences"):
        CorridorMultiInputDataset(
            [sample],
            [224, 224],
            {"sequence_length": 1, "use_depth": True},
        )


def test_temporal_dataset_uses_frame_stride_and_sequence_step(tmp_path):
    samples = []
    for index in range(13):
        image_path = tmp_path / f"stride_{index}.png"
        Image.new("RGB", (224, 224)).save(image_path)
        samples.append(
            CorridorSample(
                image_path=str(image_path),
                class_index=index % 8,
                session_name="session",
                stamp=1.0 + index * 0.25,
            )
        )
    dataset = CorridorMultiInputDataset(
        samples,
        [224, 224],
        {
            "sequence_length": 3,
            "frame_stride": 4,
            "maximum_gap_seconds": 0.4,
            "use_depth": False,
        },
        sequence_step=4,
    )

    assert len(dataset) == 2
    assert [sample.class_index for sample in dataset.sequences[0]] == [0, 4, 0]
    assert [sample.class_index for sample in dataset.sequences[1]] == [4, 0, 4]


def test_horizontal_flip_remaps_directional_label(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (224, 224)).save(image_path)
    sample = CorridorSample(str(image_path), 2, "session", stamp=1.0)
    dataset = CorridorMultiInputDataset(
        [sample],
        [224, 224],
        {"sequence_length": 1, "use_depth": False},
        augmentation_config={"horizontal_flip_probability": 1.0},
    )

    _, label = dataset[0]

    assert label == 3


def test_rgb_modality_dropout_uses_neutral_normalized_input(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (224, 224), color=(255, 255, 255)).save(image_path)
    sample = CorridorSample(str(image_path), 0, "session", stamp=1.0)
    dataset = CorridorMultiInputDataset(
        [sample],
        [224, 224],
        {"sequence_length": 1, "use_depth": False},
        augmentation_config={"rgb_dropout_probability": 1.0},
    )

    inputs, _ = dataset[0]

    assert torch.count_nonzero(inputs["rgb"]) == 0


def test_rgb_modality_dropout_probability_is_validated(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (224, 224)).save(image_path)
    sample = CorridorSample(str(image_path), 0, "session", stamp=1.0)

    with pytest.raises(ValueError, match="rgb_dropout_probability"):
        CorridorMultiInputDataset(
            [sample],
            [224, 224],
            {"sequence_length": 1, "use_depth": False},
            augmentation_config={"rgb_dropout_probability": 1.1},
        )


def test_temporal_augmentation_uses_one_rgb_and_depth_transform(tmp_path):
    samples = []
    image = Image.new("RGB", (224, 224), color=(180, 80, 30))
    depth = np.full((224, 224), 2.0, dtype=np.float32)
    for index in range(3):
        image_path = tmp_path / f"consistent_{index}.png"
        depth_path = tmp_path / f"consistent_{index}.npy"
        image.save(image_path)
        np.save(depth_path, depth)
        samples.append(
            CorridorSample(
                image_path=str(image_path),
                depth_path=str(depth_path),
                class_index=0,
                session_name="session",
                stamp=1.0 + index * 0.25,
            )
        )
    dataset = CorridorMultiInputDataset(
        samples,
        [224, 224],
        {
            "sequence_length": 3,
            "use_depth": True,
            "depth_min_m": 0.1,
            "depth_max_m": 10.0,
        },
        augmentation_config={
            "sequence_consistent": True,
            "color_jitter": {
                "brightness": 0.5,
                "contrast": 0.5,
                "saturation": 0.5,
                "hue": 0.1,
            },
            "grayscale_probability": 0.5,
            "blur_probability": 0.5,
            "depth_scale_jitter": 0.5,
        },
    )

    inputs, _ = dataset[0]

    assert torch.equal(inputs["rgb"][0], inputs["rgb"][1])
    assert torch.equal(inputs["rgb"][1], inputs["rgb"][2])
    assert torch.equal(inputs["depth"][0], inputs["depth"][1])
    assert torch.equal(inputs["depth"][1], inputs["depth"][2])


def test_passage_direction_targets_mask_turning_samples(tmp_path):
    class_names = [
        "straight_road",
        "dead_end",
        "corner_right",
        "corner_left",
        "cross_road",
        "3_way_right",
        "3_way_center",
        "3_way_left",
        "turning",
    ]
    image_path = tmp_path / "turning.png"
    Image.new("RGB", (224, 224)).save(image_path)
    dataset = CorridorMultiInputDataset(
        [CorridorSample(str(image_path), 8, "session", stamp=1.0)],
        [224, 224],
        {
            "sequence_length": 1,
            "use_depth": False,
            "output_mode": "passage_directions",
            "class_names": class_names,
            "turning_class_name": "turning",
        },
    )

    _, target = dataset[0]

    assert target["direction_mask"] == 0.0
    assert target["turning"] == 1.0
