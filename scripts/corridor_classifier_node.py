#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from scenario_navigation_msgs.msg import cmd_dir_intersection
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float32MultiArray

from corridor_classifier.config import load_config, package_root, resolve_path
from corridor_classifier.direction_debouncer import ConsecutiveConfirmDebouncer
from corridor_classifier.image_subscriber import LatestImageSubscriber
from corridor_classifier.messages import (
    make_direction_passage_message,
    make_passage_message,
)
from corridor_classifier.models import CorridorPredictor
from corridor_classifier.passage_directions import class_name_from_directions
from corridor_classifier.scenario_target_labels import ScenarioTargetLabelsSubscriber
from corridor_classifier.synchronized_subscriber import (
    LatestRgbDepthSubscriber,
)
from corridor_classifier.turning_gate import CmdDirTurningGate
from corridor_classifier.visualization import make_label_image_message


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
    turning_class_name = str(model_config.get("turning_class_name", "turning"))
    turning_index = classifier.class_names.index(turning_class_name)
    # A raw prediction must be seen this many consecutive frames before it
    # can switch the published value at all (even right after a turn resets
    # the debouncer), so a single noisy frame can never alone become the
    # published value.
    min_confirm_frames = int(rospy.get_param("~direction_min_confirm_frames", 3))
    debouncer = (
        ConsecutiveConfirmDebouncer(
            initial=(False, False, False),
            min_confirm_frames=min_confirm_frames,
        )
        if classifier.output_mode == "passage_directions"
        else ConsecutiveConfirmDebouncer(
            initial=(0,),
            min_confirm_frames=min_confirm_frames,
        )
    )
    turning_gate = CmdDirTurningGate(
        cmd_dir_topic=str(
            rospy.get_param("~cmd_dir_topic", "/cmd_dir_intersection")
        ),
        cmd_vel_topic=str(rospy.get_param("~cmd_vel_topic", "/cmd_vel")),
        # 0.20 was too high to ever trigger under vnm_ros/CARE driving:
        # measured /cmd_vel.angular.z peaked around 0.10-0.15 rad/s during
        # real turns there (vs. scenario_navigation's more abrupt, larger
        # commanded turns). 0.20 matches the threshold already used to
        # label "turning" when building the training dataset (see
        # config/dataset.yaml's turn_detection.angular_speed_threshold_rad_s),
        # so runtime and training agree on what counts as turning.
        threshold_rad_s=float(
            rospy.get_param("~turning_angular_speed_threshold_rad_s", 0.20)
        ),
        stale_timeout_seconds=float(
            rospy.get_param("~turning_stale_timeout_seconds", 1.0)
        ),
    )
    # Empty by default (disabled): only set when running alongside
    # scenario_navigation, so a raw prediction matching the destination the
    # active turn step leads into can be published immediately instead of
    # "turning" while still mid-turn (see scenario_target_labels.py).
    scenario_target_labels = ScenarioTargetLabelsSubscriber(
        topic=str(rospy.get_param("~scenario_target_labels_topic", "")),
        stale_timeout_seconds=float(
            rospy.get_param("~scenario_target_labels_stale_timeout_seconds", 1.0)
        ),
    )
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
    visualization_publisher = rospy.Publisher(
        topics.get("visualization_topic", "/corridor_classifier/visualization"),
        RosImage,
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
        turning = turning_gate.is_turning()
        if turning:
            # Discard hysteresis built up before/during the turn: a raw
            # prediction made while turning must never leak into the
            # stable output once the turn ends, and the corridor shape on
            # the other side of a turn is unrelated to it anyway.
            debouncer.reset()

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

        # Inference keeps running while turning (instead of being skipped
        # entirely) so probabilities/visualization keep showing what the
        # model actually sees, and the temporal buffer (GRU architectures)
        # stays warm instead of needing to refill from scratch the moment
        # the turn ends. /passage_type itself still always reports
        # "turning" below -- the model is not trained to classify passage
        # shape mid-turn (see README.md), so raw turn-time predictions are
        # never treated as the classification result, only shown as-is for
        # visibility.
        prediction = classifier.predict(
            PILImage.fromarray(rgb_image),
            depth_meters=depth_image,
            stamp=image_msg.header.stamp.to_sec(),
        )
        if prediction is None:
            if turning:
                passage_publisher.publish(
                    make_passage_message(turning_index, classifier.class_names)
                )
                probabilities_publisher.publish(Float32MultiArray(data=[]))
                visualization_publisher.publish(
                    make_label_image_message(turning_class_name, bridge)
                )
            rospy.loginfo_throttle(
                2.0,
                "collecting temporal context: %d/%d",
                classifier.context_length,
                classifier.required_context_length,
            )
            rate.sleep()
            continue
        if classifier.output_mode == "passage_directions":
            if turning:
                # If the turn step's destination is already visible and
                # recognized, publish it now instead of "turning" -- no
                # need to wait for cmd_vel to settle back down first.
                # scenario_navigation's turnFinish() still requires
                # turning_observed_this_step_ (a genuine turn already
                # underway) before it will act on this, so a turn that
                # never really happened still cannot complete the step.
                reached_destination = scenario_target_labels.contains(
                    prediction.class_name
                )
                passage_publisher.publish(
                    make_passage_message(
                        classifier.class_names.index(prediction.class_name)
                        if reached_destination
                        else turning_index,
                        classifier.class_names,
                    )
                )
                probabilities_publisher.publish(
                    Float32MultiArray(
                        data=list(prediction.direction_probabilities)
                    )
                )
                visualization_publisher.publish(
                    make_label_image_message(
                        prediction.class_name
                        if reached_destination
                        else f"{prediction.class_name} (turning)",
                        bridge,
                    )
                )
                rospy.loginfo_throttle(
                    1.0,
                    "corridor(raw,turning,reached_destination=%s)=%s "
                    "open(raw)=%s probabilities=(%.3f,%.3f,%.3f)",
                    reached_destination,
                    prediction.class_name,
                    prediction.open_directions,
                    *prediction.direction_probabilities,
                )
                rate.sleep()
                continue
            stable_directions = tuple(
                debouncer.update(
                    tuple(bool(v) for v in prediction.open_directions)
                )
            )
            passage_publisher.publish(
                make_direction_passage_message(
                    stable_directions,
                    classifier.class_names,
                )
            )
            probabilities_publisher.publish(
                Float32MultiArray(
                    data=list(prediction.direction_probabilities)
                )
            )
            visualization_publisher.publish(
                make_label_image_message(
                    class_name_from_directions(stable_directions), bridge
                )
            )
            rospy.loginfo(
                "corridor(stable,published)=%s corridor(raw,per-frame)=%s "
                "open(front,left,right)(stable)=%s open(raw)=%s "
                "probabilities=(%.3f,%.3f,%.3f)",
                class_name_from_directions(stable_directions),
                prediction.class_name,
                stable_directions,
                prediction.open_directions,
                *prediction.direction_probabilities,
            )
        else:
            if turning:
                raw_class_name = classifier.class_names[prediction.class_index]
                # See the passage_directions branch above for why this
                # bypasses "turning" -- same idea, just class-mode.
                reached_destination = scenario_target_labels.contains(
                    raw_class_name
                )
                passage_publisher.publish(
                    make_passage_message(
                        prediction.class_index
                        if reached_destination
                        else turning_index,
                        classifier.class_names,
                    )
                )
                probabilities_publisher.publish(
                    Float32MultiArray(data=list(prediction.probabilities))
                )
                visualization_publisher.publish(
                    make_label_image_message(
                        raw_class_name
                        if reached_destination
                        else f"{raw_class_name} (turning)",
                        bridge,
                    )
                )
                rospy.loginfo_throttle(
                    1.0,
                    "corridor(raw,turning,reached_destination=%s)=%s confidence=%.3f",
                    reached_destination,
                    raw_class_name,
                    prediction.confidence,
                )
                rate.sleep()
                continue
            (stable_class_index,) = debouncer.update((prediction.class_index,))
            passage_publisher.publish(
                make_passage_message(
                    stable_class_index,
                    classifier.class_names,
                )
            )
            probabilities_publisher.publish(
                Float32MultiArray(data=list(prediction.probabilities))
            )
            visualization_publisher.publish(
                make_label_image_message(
                    classifier.class_names[stable_class_index], bridge
                )
            )
            rospy.loginfo(
                "corridor=%s raw=%s confidence=%.3f",
                classifier.class_names[stable_class_index],
                classifier.class_names[prediction.class_index],
                prediction.confidence,
            )
        rate.sleep()


if __name__ == "__main__":
    main()
