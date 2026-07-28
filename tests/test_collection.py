import csv

from PIL import Image

from corridor_classifier.collection import (
    DatasetSessionWriter,
    class_index_from_one_hot,
)


CLASS_NAMES = (
    "straight_road",
    "dead_end",
    "corner_right",
    "corner_left",
    "cross_road",
    "3_way_right",
    "3_way_center",
    "3_way_left",
)


def test_one_hot_label_validation():
    assert class_index_from_one_hot([0, 0, 1, 0, 0, 0, 0, 0], 8) == 2
    assert class_index_from_one_hot([0] * 8, 8) is None
    assert class_index_from_one_hot([1, 1, 0, 0, 0, 0, 0, 0], 8) is None
    assert class_index_from_one_hot([1, 0, 0, 0], 8) is None


def test_session_writer_resizes_image_and_writes_manifest(tmp_path):
    session_dir = tmp_path / "session"
    writer = DatasetSessionWriter(
        session_dir=str(session_dir),
        class_names=CLASS_NAMES,
        input_size=[224, 224],
        image_format="jpg",
        jpeg_quality=95,
        metadata={"dataset_type": "train"},
    )
    output_path = writer.save(
        Image.new("RGB", (640, 480), color=(10, 20, 30)),
        class_index=5,
        stamp=12.5,
    )
    writer.close()

    with Image.open(output_path) as image:
        assert image.size == (224, 224)
    with open(session_dir / "samples.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["class_index"] == "5"
    assert rows[0]["class_name"] == "3_way_right"

