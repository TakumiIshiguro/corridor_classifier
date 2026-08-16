#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from scenario_navigation_msgs.msg import cmd_dir_intersection
from std_msgs.msg import Float32MultiArray

from corridor_classifier.config import load_config, package_root, resolve_path
from corridor_classifier.image_subscriber import LatestImageSubscriber
from corridor_classifier.messages import make_passage_message
from corridor_classifier.models import CorridorPredictor
from corridor_classifier.synchronized_subscriber import (
    LatestRgbDepthSubscriber,
)


def _apply_ros_overrides(config):
    model = config["model"]
    runtime = config["runtime"]

    checkpoint_override = str(
        rospy.get_param("~checkpoint_path_override", "")
    ).strip()
    if checkpoint_override:
        model["checkpoint_path"] = checkpoint_override

    device_override = str(rospy.get_param("~device_override", "")).strip()
    if device_override:
        model["device"] = device_override

    rate_override = float(rospy.get_param("~inference_rate_override", 0.0))
    if rate_override > 0.0:
        runtime["inference_rate"] = rate_override


def main():
    rospy.init_node("corridor_classifier")
    config_dir = rospy.get_param("~config_dir", None)
    config = load_config(config_dir)
    _apply_ros_overrides(config)

    model_config = config["model"]
    runtime = config["runtime"]
    topics = config["topics"]
    checkpoint_path = resolve_path(
        model_config["checkpoint_path"],
        package_root(),
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "corridor classifier checkpoint was not found: "
            f"{checkpoint_path}. See weights/README.md."
        )

    classifier = CorridorPredictor(model_config, checkpoint_path)
    bridge = CvBridge()
    if classifier.use_depth:
        subscriber = LatestRgbDepthSubscriber(
            topics["image_topic"], topics["depth_topic"]
        )
    else:
        subscriber = LatestImageSubscriber(topics["image_topic"])
    passage_publisher = rospy.Publisher(
        topics["passage_type_topic"],
        cmd_dir_intersection,
        queue_size=1,
    )
    probabilities_publisher = rospy.Publisher(
        topics["probabilities_topic"],
        Float32MultiArray,
        queue_size=1,
    )

    rate_hz = float(runtime["inference_rate"])
    rate = rospy.Rate(rate_hz)
    rospy.loginfo(
        "corridor_classifier loaded architecture=%s backbone=%s from %s on %s "
        "(input=%sx%s, sequence=%d, stride=%d, depth=%s, rate=%.2f Hz)",
        model_config["architecture"],
        model_config["model_name"],
        checkpoint_path,
        classifier.device,
        model_config["input_size"][0],
        model_config["input_size"][1],
        classifier.sequence_length,
        classifier.frame_stride,
        classifier.use_depth,
        rate_hz,
    )

    while not rospy.is_shutdown():
        received = subscriber.take_latest()
        if received is None:
            rate.sleep()
            continue

        if classifier.use_depth:
            image_msg, depth_msg = received
        else:
            image_msg = received
            depth_msg = None

        try:
            rgb_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
            depth_image = (
                bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
                if depth_msg is not None
                else None
            )
        except CvBridgeError as error:
            rospy.logwarn_throttle(
                5.0,
                f"failed to convert camera image: {error}",
            )
            rate.sleep()
            continue

        prediction = classifier.predict(
            PILImage.fromarray(rgb_image),
            depth_meters=depth_image,
            stamp=image_msg.header.stamp.to_sec(),
        )
        if prediction is None:
            rospy.loginfo_throttle(
                2.0,
                "collecting temporal context: %d/%d",
                classifier.context_length,
                classifier.required_context_length,
            )
            rate.sleep()
            continue
        passage_publisher.publish(
            make_passage_message(
                prediction.class_index,
                classifier.class_names,
            )
        )
        probabilities_publisher.publish(
            Float32MultiArray(data=list(prediction.probabilities))
        )
        rospy.loginfo(
            "corridor=%s confidence=%.3f",
            classifier.class_names[prediction.class_index],
            prediction.confidence,
        )
        rate.sleep()


if __name__ == "__main__":
    main()
