import numpy as np
import pytest
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
