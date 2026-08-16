from types import SimpleNamespace

from corridor_classifier.bag_collection import (
    image_stamp,
    iter_labeled_images,
)


class FakeTime:
    def __init__(self, seconds):
        self._seconds = float(seconds)

    def to_sec(self):
        return self._seconds


def make_image(header_stamp):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=FakeTime(header_stamp))
    )


def make_label(class_index):
    values = [0] * 8
    if class_index is not None:
        values[class_index] = 1
    return SimpleNamespace(intersection_label=values)


def test_bag_images_use_latest_fresh_label_and_sample_interval():
    image_1 = make_image(101.0)
    image_2 = make_image(101.1)
    image_3 = make_image(101.3)
    stale_image = make_image(102.0)
    messages = [
        ("/label", make_label(2), FakeTime(1.0)),
        ("/image", image_1, FakeTime(1.0)),
        ("/image", image_2, FakeTime(1.1)),
        ("/image", image_3, FakeTime(1.3)),
        ("/image", stale_image, FakeTime(2.0)),
    ]

    samples = list(
        iter_labeled_images(
            messages,
            image_topic="/image",
            label_topic="/label",
            num_classes=8,
            sample_dt=0.25,
            label_timeout=0.5,
        )
    )

    assert samples == [
        (image_1, 2, 101.0),
        (image_3, 2, 101.3),
    ]


def test_invalid_label_clears_previous_label():
    image_before = make_image(11.0)
    image_after = make_image(12.0)
    messages = [
        ("/label", make_label(4), FakeTime(1.0)),
        ("/image", image_before, FakeTime(1.1)),
        ("/label", make_label(None), FakeTime(1.2)),
        ("/image", image_after, FakeTime(1.3)),
    ]

    samples = list(
        iter_labeled_images(
            messages,
            image_topic="/image",
            label_topic="/label",
            num_classes=8,
            sample_dt=0.01,
            label_timeout=1.0,
        )
    )

    assert samples == [(image_before, 4, 11.0)]


def test_image_stamp_falls_back_to_bag_time():
    assert image_stamp(make_image(0.0), FakeTime(9.5)) == 9.5


def test_bag_class_override_can_assign_turning_label():
    image = make_image(11.0)
    messages = [
        ("/label", make_label(3), FakeTime(1.0)),
        ("/image", image, FakeTime(1.1)),
    ]

    samples = list(
        iter_labeled_images(
            messages,
            image_topic="/image",
            label_topic="/label",
            num_classes=8,
            sample_dt=0.25,
            label_timeout=0.5,
            class_index_override=lambda stamp, source_class: 8,
        )
    )

    assert samples == [(image, 8, 11.0)]
