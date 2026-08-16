import csv

from PIL import Image
import yaml

from corridor_classifier.collection import (
    bridge_turning_gaps_in_session,
    DatasetSessionWriter,
    class_index_from_one_hot,
    replace_post_turn_labels_in_session,
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


def test_post_turn_session_replacement_updates_metadata(tmp_path):
    session_dir = tmp_path / "session"
    class_names = CLASS_NAMES + ("turning",)
    writer = DatasetSessionWriter(
        session_dir=str(session_dir),
        class_names=class_names,
        input_size=[8, 8],
        image_format="png",
        jpeg_quality=95,
        metadata={"dataset_type": "train"},
    )
    for index, class_index in enumerate((3, 8, 3, 0)):
        writer.save(Image.new("RGB", (8, 8)), class_index, index * 0.25)
    writer.close()

    changed = replace_post_turn_labels_in_session(
        str(session_dir), class_names, 8, 6.0
    )

    with open(session_dir / "samples.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with open(session_dir / "metadata.yaml") as stream:
        metadata = yaml.safe_load(stream)
    assert changed == 1
    assert [int(row["class_index"]) for row in rows] == [3, 8, 0, 0]
    assert metadata["post_turn_next_label"]["changed_samples"] == 1


def test_turning_gap_bridge_updates_session_manifest(tmp_path):
    session_dir = tmp_path / "session"
    class_names = CLASS_NAMES + ("turning",)
    writer = DatasetSessionWriter(
        session_dir=str(session_dir),
        class_names=class_names,
        input_size=[8, 8],
        image_format="png",
        jpeg_quality=95,
        metadata={"dataset_type": "train"},
    )
    for index, class_index in enumerate((8, 0, 8)):
        writer.save(Image.new("RGB", (8, 8)), class_index, index * 0.25)
    writer.close()

    changed = bridge_turning_gaps_in_session(
        str(session_dir), class_names, 8, 1.5
    )

    with open(session_dir / "samples.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert changed == 1
    assert [int(row["class_index"]) for row in rows] == [8, 8, 8]
