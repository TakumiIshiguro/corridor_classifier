from typing import Iterable, Iterator, Optional, Tuple

from corridor_classifier.collection import class_index_from_one_hot


def time_to_sec(stamp) -> float:
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    return float(stamp)


def image_stamp(image_msg, bag_time) -> float:
    header = getattr(image_msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        stamp_sec = time_to_sec(stamp)
        if stamp_sec > 0.0:
            return stamp_sec
    return time_to_sec(bag_time)


def iter_labeled_images(
    messages: Iterable[Tuple[str, object, object]],
    image_topic: str,
    label_topic: str,
    num_classes: int,
    sample_dt: float,
    label_timeout: float,
) -> Iterator[Tuple[object, int, float]]:
    """Yield bag images paired with the latest valid preceding label.

    Label freshness and sampling are evaluated using rosbag record time. The
    image header stamp is retained for the dataset manifest when it is valid.
    """
    current_class: Optional[int] = None
    label_received_at: Optional[float] = None
    last_saved_at = float("-inf")

    for topic, msg, bag_time in messages:
        event_time = time_to_sec(bag_time)
        if topic == label_topic:
            current_class = class_index_from_one_hot(
                msg.intersection_label,
                num_classes,
            )
            label_received_at = event_time
            continue
        if topic != image_topic:
            continue
        if current_class is None or label_received_at is None:
            continue

        label_age = event_time - label_received_at
        if label_age < 0.0 or label_age > label_timeout:
            continue
        if event_time - last_saved_at < sample_dt:
            continue

        yield msg, current_class, image_stamp(msg, bag_time)
        last_saved_at = event_time
