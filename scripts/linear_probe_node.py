#!/usr/bin/env python3
"""ROS inference node for the closed-form ridge linear-probe corridor
classifier (see src/corridor_classifier/linear_probe.py).

Unlike corridor_classifier_node.py, this node is stateless across frames
(single-frame classification, no GRU temporal buffer to fill) and, for the
production checkpoint, does not use depth, so it only subscribes to the
camera image topic.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from scenario_navigation_msgs.msg import cmd_dir_intersection
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float32MultiArray

from corridor_classifier.direction_debouncer import ConsecutiveConfirmDebouncer
from corridor_classifier.image_subscriber import LatestImageSubscriber
from corridor_classifier.linear_probe import RidgeLinearProbePredictor
from corridor_classifier.messages import make_direction_passage_message, make_passage_message
from corridor_classifier.passage_directions import class_name_from_directions
from corridor_classifier.scenario_target_labels import ScenarioTargetLabelsSubscriber
from corridor_classifier.synchronized_subscriber import LatestRgbDepthSubscriber
from corridor_classifier.turning_gate import CmdDirTurningGate
from corridor_classifier.visualization import make_label_image_message


def main():
    rospy.init_node("corridor_classifier_linear_probe")
    checkpoint_path = str(rospy.get_param("~checkpoint_path"))
    device = str(rospy.get_param("~device_override", "auto"))
    image_topic = str(rospy.get_param("~image_topic", "/camera_center/image_raw"))
    depth_topic = str(rospy.get_param("~depth_topic", "/unidepth/depth"))
    passage_type_topic = str(rospy.get_param("~passage_type_topic", "/passage_type"))
    probabilities_topic = str(
        rospy.get_param("~probabilities_topic", "/corridor_classifier/probabilities")
    )
    visualization_topic = str(
        rospy.get_param("~visualization_topic", "/corridor_classifier/visualization")
    )
    rate_hz = float(rospy.get_param("~inference_rate", 4.0))
    cmd_dir_topic = str(rospy.get_param("~cmd_dir_topic", "/cmd_dir_intersection"))
    cmd_vel_topic = str(rospy.get_param("~cmd_vel_topic", "/cmd_vel"))
    # 0.20 was too high to ever trigger under vnm_ros/CARE driving:
    # measured /cmd_vel.angular.z peaked around 0.10-0.15 rad/s during real
    # turns there (vs. scenario_navigation's more abrupt, larger commanded
    # turns). 0.20 matches the threshold already used to label "turning"
    # when building the training dataset (see config/dataset.yaml's
    # turn_detection.angular_speed_threshold_rad_s), so runtime and
    # training agree on what counts as turning.
    turning_threshold = float(
        rospy.get_param("~turning_angular_speed_threshold_rad_s", 0.20)
    )
    turning_stale_timeout = float(
        rospy.get_param("~turning_stale_timeout_seconds", 1.0)
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"linear probe checkpoint was not found: {checkpoint_path}")

    predictor = RidgeLinearProbePredictor(checkpoint_path, device=device)
    turning_index = predictor.class_names.index("turning")
    # A raw prediction must be seen this many consecutive frames before it
    # can switch the published value at all (even right after a turn resets
    # the debouncer), so a single noisy frame can never alone become the
    # published value.
    min_confirm_frames = int(rospy.get_param("~direction_min_confirm_frames", 3))
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(False, False, False),
        min_confirm_frames=min_confirm_frames,
    )
    turning_gate = CmdDirTurningGate(
        cmd_dir_topic=cmd_dir_topic,
        cmd_vel_topic=cmd_vel_topic,
        threshold_rad_s=turning_threshold,
        stale_timeout_seconds=turning_stale_timeout,
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
    if predictor.use_depth:
        subscriber = LatestRgbDepthSubscriber(image_topic, depth_topic)
    else:
        subscriber = LatestImageSubscriber(image_topic)

    passage_publisher = rospy.Publisher(passage_type_topic, cmd_dir_intersection, queue_size=1)
    probabilities_publisher = rospy.Publisher(
        probabilities_topic, Float32MultiArray, queue_size=1
    )
    visualization_publisher = rospy.Publisher(
        visualization_topic, RosImage, queue_size=1
    )

    rate = rospy.Rate(rate_hz)
    rospy.loginfo(
        "corridor_classifier_linear_probe loaded readout=%s depth=%s from %s on %s "
        "(input=%sx%s, rate=%.2f Hz)",
        predictor.dino_readout,
        predictor.use_depth,
        checkpoint_path,
        predictor.device,
        predictor.input_size[0],
        predictor.input_size[1],
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

        if predictor.use_depth:
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
            rospy.logwarn_throttle(5.0, f"failed to convert camera image: {error}")
            rate.sleep()
            continue

        # Inference keeps running while turning (instead of being skipped
        # entirely) so probabilities/visualization keep showing what the
        # model actually sees. /passage_type itself still always reports
        # "turning" below -- the model is not trained to classify passage
        # shape mid-turn (see README.md), so raw turn-time predictions are
        # never treated as the classification result, only shown as-is for
        # visibility.
        prediction = predictor.predict(PILImage.fromarray(rgb_image), depth_meters=depth_image)
        if turning:
            # If the turn step's destination is already visible and
            # recognized, publish it now instead of "turning" -- no need
            # to wait for cmd_vel to settle back down first.
            # scenario_navigation's turnFinish() still requires
            # turning_observed_this_step_ (a genuine turn already
            # underway) before it will act on this, so a turn that never
            # really happened still cannot complete the step.
            reached_destination = scenario_target_labels.contains(
                prediction.class_name
            )
            passage_publisher.publish(
                make_passage_message(
                    predictor.class_names.index(prediction.class_name)
                    if reached_destination
                    else turning_index,
                    predictor.class_names,
                )
            )
            probabilities_publisher.publish(
                Float32MultiArray(data=list(prediction.direction_scores))
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
                "corridor(raw,turning,reached_destination=%s)=%s open(raw)=%s scores=(%.3f,%.3f,%.3f)",
                reached_destination,
                prediction.class_name,
                prediction.open_directions,
                *prediction.direction_scores,
            )
            rate.sleep()
            continue

        stable_directions = tuple(
            debouncer.update(tuple(bool(v) for v in prediction.open_directions))
        )
        passage_publisher.publish(
            make_direction_passage_message(stable_directions, predictor.class_names)
        )
        probabilities_publisher.publish(
            Float32MultiArray(data=list(prediction.direction_scores))
        )
        visualization_publisher.publish(
            make_label_image_message(
                class_name_from_directions(stable_directions), bridge
            )
        )
        rospy.loginfo(
            "corridor=%s open(front,left,right)=%s raw=%s scores=(%.3f,%.3f,%.3f)",
            prediction.class_name,
            stable_directions,
            prediction.open_directions,
            *prediction.direction_scores,
        )
        rate.sleep()


if __name__ == "__main__":
    main()
