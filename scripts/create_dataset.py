#!/usr/bin/env python3
import os
import sys
from datetime import datetime
from threading import Lock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
import rosbag
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from scenario_navigation_msgs.msg import cmd_dir_intersection

from corridor_classifier.collection import (
    bridge_turning_gaps_in_session,
    DatasetSessionWriter,
    class_index_from_one_hot,
    replace_post_turn_labels_in_session,
)
from corridor_classifier.bag_collection import iter_labeled_images
from corridor_classifier.config import (
    load_collection_config,
    package_root,
    resolve_path,
)
from corridor_classifier.image_subscriber import LatestImageSubscriber
from corridor_classifier.turn_detection import detector_from_pose_messages


class LatestLabel:
    def __init__(self, num_classes):
        self._num_classes = int(num_classes)
        self._lock = Lock()
        self._class_index = None
        self._received_at = None

    def callback(self, msg):
        class_index = class_index_from_one_hot(
            msg.intersection_label,
            self._num_classes,
        )
        with self._lock:
            self._class_index = class_index
            self._received_at = rospy.get_time()

    def get(self, now, timeout):
        with self._lock:
            if self._class_index is None or self._received_at is None:
                return None
            if now - self._received_at > timeout or now < self._received_at:
                return None
            return self._class_index


def _apply_overrides(config):
    collection = config["collection"]
    dataset_dir = str(
        rospy.get_param("~dataset_dir_override", "")
    ).strip()
    if dataset_dir:
        config["paths"]["dataset_dir"] = dataset_dir

    bag_path = str(rospy.get_param("~bag_path_override", "")).strip()
    if bag_path:
        collection["bag_path"] = bag_path

    dataset_type = str(
        rospy.get_param("~dataset_type_override", "")
    ).strip()
    if dataset_type:
        if dataset_type not in ("train", "test"):
            raise ValueError(
                "dataset_type_override must be train or test"
            )
        collection["dataset_type"] = dataset_type

    session_name = str(
        rospy.get_param("~session_name_override", "")
    ).strip()
    if session_name:
        collection["session_name"] = session_name


def _bag_topics(bag):
    topic_info = bag.get_type_and_topic_info()
    topics = getattr(topic_info, "topics", None)
    if topics is None:
        topics = topic_info[1]
    return set(topics.keys())


def _validate_bag_topics(bag, image_topic, label_topic, additional_topics=()):
    available_topics = _bag_topics(bag)
    missing_topics = [
        topic
        for topic in (image_topic, label_topic, *additional_topics)
        if topic not in available_topics
    ]
    if missing_topics:
        raise ValueError(
            "rosbag is missing required topic(s): "
            + ", ".join(missing_topics)
        )


def _collect_live(writer, model, topics, collection, bridge):
    if collection["turn_detection"]["enabled"]:
        raise ValueError(
            "centered-window turn detection currently requires collection.source=bag"
        )
    image_subscriber = LatestImageSubscriber(topics["image_topic"])
    latest_label = LatestLabel(
        collection["turn_detection"]["source_num_classes"]
    )
    label_subscriber = rospy.Subscriber(
        topics["label_topic"],
        cmd_dir_intersection,
        latest_label.callback,
        queue_size=1,
    )

    sample_dt = float(collection["sample_dt"])
    label_timeout = float(collection["label_timeout"])
    last_saved_at = float("-inf")
    rate = rospy.Rate(max(10.0, 2.0 / sample_dt))

    try:
        while not rospy.is_shutdown():
            now = rospy.get_time()
            if now - last_saved_at < sample_dt:
                rate.sleep()
                continue

            class_index = latest_label.get(now, label_timeout)
            if class_index is None:
                rate.sleep()
                continue
            image_msg = image_subscriber.take_latest()
            if image_msg is None:
                rate.sleep()
                continue

            try:
                rgb_image = bridge.imgmsg_to_cv2(
                    image_msg,
                    desired_encoding="rgb8",
                )
            except CvBridgeError as error:
                rospy.logwarn_throttle(
                    5.0,
                    f"failed to convert camera image: {error}",
                )
                rate.sleep()
                continue

            stamp = image_msg.header.stamp.to_sec()
            if stamp <= 0.0:
                stamp = now
            writer.save(
                PILImage.fromarray(rgb_image),
                class_index,
                stamp,
            )
            last_saved_at = now
            rospy.loginfo_throttle(
                5.0,
                "saved %d samples, latest_class=%s",
                writer.sample_count,
                model["class_names"][class_index],
            )
            rate.sleep()
    finally:
        label_subscriber.unregister()


def _collect_bag(writer, bag_path, model, topics, collection, bridge):
    image_topic = topics["image_topic"]
    label_topic = topics["label_topic"]
    turn_detection = collection["turn_detection"]
    with rosbag.Bag(bag_path, "r") as bag:
        extra_topics = (
            [turn_detection["pose_topic"]]
            if turn_detection["enabled"]
            else []
        )
        _validate_bag_topics(
            bag,
            image_topic,
            label_topic,
            additional_topics=extra_topics,
        )
        turning_detector = None
        if turn_detection["enabled"]:
            turning_detector = detector_from_pose_messages(
                bag.read_messages(topics=[turn_detection["pose_topic"]]),
                angular_speed_threshold_rad_s=turn_detection[
                    "angular_speed_threshold_rad_s"
                ],
                window_seconds=turn_detection["window_seconds"],
                minimum_duration_seconds=turn_detection[
                    "minimum_duration_seconds"
                ],
                padding_seconds=turn_detection["padding_seconds"],
                maximum_pose_gap_seconds=turn_detection[
                    "maximum_pose_gap_seconds"
                ],
            )
            rospy.loginfo(
                "detected %d turning intervals from %s",
                len(turning_detector.intervals),
                turn_detection["pose_topic"],
            )
        messages = bag.read_messages(topics=[image_topic, label_topic])
        class_index_override = None
        if turning_detector is not None:
            turn_class_index = int(turn_detection["class_index"])

            def class_index_override(event_time, source_class_index):
                if turning_detector.is_turning(event_time):
                    return turn_class_index
                return source_class_index

        samples = iter_labeled_images(
            messages=messages,
            image_topic=image_topic,
            label_topic=label_topic,
            num_classes=turn_detection["source_num_classes"],
            sample_dt=float(collection["sample_dt"]),
            label_timeout=float(collection["label_timeout"]),
            class_index_override=class_index_override,
        )
        for image_msg, class_index, stamp in samples:
            if rospy.is_shutdown():
                break
            try:
                rgb_image = bridge.imgmsg_to_cv2(
                    image_msg,
                    desired_encoding="rgb8",
                )
            except CvBridgeError as error:
                rospy.logwarn(
                    "failed to convert bag image at %.9f: %s",
                    stamp,
                    error,
                )
                continue
            writer.save(
                PILImage.fromarray(rgb_image),
                class_index,
                stamp,
            )
            if writer.sample_count % 100 == 0:
                rospy.loginfo("saved %d samples", writer.sample_count)


def main():
    rospy.init_node("corridor_dataset_collector")
    config_dir = rospy.get_param("~config_dir", None)
    config = load_collection_config(config_dir)
    _apply_overrides(config)

    model = config["model"]
    topics = config["topics"]
    collection = config["collection"]
    collection["turn_detection"] = config["turn_detection"]
    dataset_root = resolve_path(config["paths"]["dataset_dir"], package_root())
    bag_path = ""
    if collection["source"] == "bag":
        bag_path = resolve_path(collection["bag_path"], package_root())
        if not os.path.isfile(bag_path):
            raise FileNotFoundError(f"rosbag does not exist: {bag_path}")
        with rosbag.Bag(bag_path, "r") as bag:
            _validate_bag_topics(
                bag,
                topics["image_topic"],
                topics["label_topic"],
                additional_topics=(
                    [config["turn_detection"]["pose_topic"]]
                    if config["turn_detection"]["enabled"]
                    else []
                ),
            )

    session_name = collection["session_name"]
    if not session_name:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = os.path.join(
        dataset_root,
        collection["dataset_type"],
        session_name,
    )

    writer = DatasetSessionWriter(
        session_dir=session_dir,
        class_names=model["class_names"],
        input_size=model["input_size"],
        image_format=collection["image_format"],
        jpeg_quality=collection["jpeg_quality"],
        metadata={
            "source": collection["source"],
            "bag_path": bag_path,
            "dataset_type": collection["dataset_type"],
            "session_name": session_name,
            "image_topic": topics["image_topic"],
            "label_topic": topics["label_topic"],
            "sample_dt": collection["sample_dt"],
            "label_timeout": collection["label_timeout"],
            "turn_detection": config["turn_detection"],
        },
    )
    rospy.on_shutdown(writer.close)

    bridge = CvBridge()
    sample_dt = float(collection["sample_dt"])
    rospy.loginfo(
        "collecting corridor dataset from %s to %s at %.2f Hz, input=%sx%s",
        collection["source"],
        session_dir,
        1.0 / sample_dt,
        model["input_size"][0],
        model["input_size"][1],
    )

    try:
        if collection["source"] == "bag":
            _collect_bag(
                writer,
                bag_path,
                model,
                topics,
                collection,
                bridge,
            )
        else:
            _collect_live(writer, model, topics, collection, bridge)
    finally:
        writer.close()

    if collection["source"] == "bag" and config["turn_detection"]["enabled"]:
        turn_config = config["turn_detection"]
        bridged = bridge_turning_gaps_in_session(
            session_dir=session_dir,
            class_names=model["class_names"],
            turn_class_index=turn_config["class_index"],
            maximum_seconds=turn_config["turning_gap_bridge_max_seconds"],
        )
        changed = replace_post_turn_labels_in_session(
            session_dir=session_dir,
            class_names=model["class_names"],
            turn_class_index=turn_config["class_index"],
            maximum_seconds=turn_config[
                "post_turn_next_label_max_seconds"
            ],
        )
        rospy.loginfo(
            "bridged %d short turning-gap sample(s); replaced %d short "
            "post-turn stale-label sample(s) with the following label",
            bridged,
            changed,
        )

    rospy.loginfo(
        "dataset collection finished: %d samples in %s",
        writer.sample_count,
        session_dir,
    )


if __name__ == "__main__":
    main()
