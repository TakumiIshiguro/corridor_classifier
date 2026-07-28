from PIL import Image

from corridor_classifier.collection import DatasetSessionWriter
from corridor_classifier.dataset import (
    CorridorDataset,
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


def test_dataset_returns_normalized_224_tensor(tmp_path):
    make_session(tmp_path, "train", "session_0", 3)
    train_samples = load_dataset_samples(str(tmp_path / "train"), 8)
    dataset = CorridorDataset(train_samples, [224, 224])

    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert label == 3
