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
from std_msgs.msg import Float32MultiArray

from corridor_classifier.direction_debouncer import ConsecutiveConfirmDebouncer
from corridor_classifier.image_subscriber import LatestImageSubscriber
from corridor_classifier.linear_probe import RidgeLinearProbePredictor
from corridor_classifier.messages import make_direction_passage_message, make_passage_message
from corridor_classifier.scenario_target_labels import ScenarioTargetLabelsSubscriber
from corridor_classifier.synchronized_subscriber import LatestRgbDepthSubscriber
from corridor_classifier.turning_gate import CmdVelTurningGate


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
    rate_hz = float(rospy.get_param("~inference_rate", 4.0))
    cmd_vel_topic = str(rospy.get_param("~cmd_vel_topic", "/cmd_vel"))
    # 0.20 was too high to ever trigger under vnm_ros/CARE driving:
    # measured /cmd_vel.angular.z peaked around 0.10-0.15 rad/s during real
    # turns there (vs. scenario_navigation's more abrupt, larger commanded
    # turns). 0.08 leaves some margin below that measured range.
    turning_threshold = float(
        rospy.get_param("~turning_angular_speed_threshold_rad_s", 0.08)
    )
    turning_stale_timeout = float(
        rospy.get_param("~turning_stale_timeout_seconds", 1.0)
    )
    # Empty by default (disabled): only set when running alongside vnm_ros
    # CARE, so its obstacle-avoidance steering is never mistaken for a
    # scenario turn (see corridor_classifier/turning_gate.py).
    care_avoidance_topic = str(rospy.get_param("~care_avoidance_topic", ""))

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"linear probe checkpoint was not found: {checkpoint_path}")

    predictor = RidgeLinearProbePredictor(checkpoint_path, device=device)
    turning_index = predictor.class_names.index("turning")
    # A candidate value (see min_confirm_frames below) is held for this
    # many frames before another switch is accepted (see
    # direction_debouncer.py). NOTE: 32 exceeds the dataset-derived safe
    # upper bound of 31 (the shortest genuine non-turning run in
    # dataset/corridor/bags_turning, sessions a-n, that is immediately
    # followed by a turning segment), so in principle a real segment that
    # short could be held through and never reflected in the output;
    # chosen anyway as a deliberate stability/responsiveness tradeoff.
    confirm_frames = int(rospy.get_param("~direction_confirm_frames", 32))
    # A raw prediction must be seen this many consecutive frames before it
    # can switch at all (even the very first switch, e.g. right after a
    # turn resets the debouncer), so a single noisy frame can never alone
    # become the published value. It also still needs this much evidence
    # before it can bypass the hold above when it matches
    # scenario_navigation's current target (see scenario_target_labels.py
    # below) -- so a single noisy frame cannot alone make
    # scenario_navigation advance a step, either. Deliberately much
    # smaller than direction_confirm_frames: the point is rejecting
    # single-frame noise, not making transitions wait as long as the hold.
    min_confirm_frames = int(rospy.get_param("~direction_min_confirm_frames", 3))
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(False, False, False),
        confirm_frames=confirm_frames,
        min_confirm_frames=min_confirm_frames,
    )
    turning_gate = CmdVelTurningGate(
        topic=cmd_vel_topic,
        threshold_rad_s=turning_threshold,
        stale_timeout_seconds=turning_stale_timeout,
        care_avoidance_topic=care_avoidance_topic,
    )
    # Empty by default (disabled): only set when running alongside
    # scenario_navigation, so a raw prediction matching its current target
    # can bypass the debounce hold immediately instead of risking the
    # target being missed or delayed (see scenario_target_labels.py).
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
        if turning_gate.is_turning():
            # Discard hysteresis built up before the turn: the corridor
            # shape on the other side of a turn is unrelated to it.
            debouncer.reset()
            passage_publisher.publish(
                make_passage_message(turning_index, predictor.class_names)
            )
            probabilities_publisher.publish(Float32MultiArray(data=[]))
            rospy.loginfo_throttle(1.0, "corridor=turning (cmd_vel turning)")
            rate.sleep()
            continue

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

        prediction = predictor.predict(PILImage.fromarray(rgb_image), depth_meters=depth_image)
        bypass_hold = scenario_target_labels.contains(prediction.class_name)
        stable_directions = tuple(
            debouncer.update(
                tuple(bool(v) for v in prediction.open_directions),
                bypass_hold=bypass_hold,
            )
        )
        passage_publisher.publish(
            make_direction_passage_message(stable_directions, predictor.class_names)
        )
        probabilities_publisher.publish(
            Float32MultiArray(data=list(prediction.direction_scores))
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
