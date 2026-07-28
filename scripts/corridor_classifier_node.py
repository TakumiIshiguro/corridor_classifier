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
from corridor_classifier.dino_classifier import DINOClassifier
from corridor_classifier.image_subscriber import LatestImageSubscriber
from corridor_classifier.messages import make_passage_message


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

    classifier = DINOClassifier(model_config, checkpoint_path)
    bridge = CvBridge()
    image_subscriber = LatestImageSubscriber(topics["image_topic"])
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
        "corridor_classifier loaded %s from %s on %s (input=%sx%s, rate=%.2f Hz)",
        model_config["model_name"],
        checkpoint_path,
        classifier.device,
        model_config["input_size"][0],
        model_config["input_size"][1],
        rate_hz,
    )

    while not rospy.is_shutdown():
        image_msg = image_subscriber.take_latest()
        if image_msg is None:
            rate.sleep()
            continue

        try:
            rgb_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        except CvBridgeError as error:
            rospy.logwarn_throttle(
                5.0,
                f"failed to convert camera image: {error}",
            )
            rate.sleep()
            continue

        prediction = classifier.predict(PILImage.fromarray(rgb_image))
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
