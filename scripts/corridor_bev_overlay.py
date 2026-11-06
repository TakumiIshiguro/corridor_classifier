#!/usr/bin/env python3
"""Publish the classifier's wall BEV with its passage class drawn on top."""
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import rospy
from cv_bridge import CvBridge
from scenario_navigation_msgs.msg import cmd_dir_intersection
from sensor_msgs.msg import Image as RosImage

from corridor_classifier.bev_overlay import render_passage_bev
from corridor_classifier.bev_points import LatestBevGridSubscriber
from corridor_classifier.config import load_config


class PassageBevOverlay:
    def __init__(self, config):
        topics = config["topics"]
        grid_config = config["bev_grid"]
        self.forward_range_m = grid_config["forward_range_m"]
        self.map_width_m = float(grid_config["map_width_m"])
        self.min_points = float(config["model"].get("bev_min_points", 0.0))
        self.binary = bool(config["model"].get("bev_binary", False))
        self.stale_seconds = float(rospy.get_param("~stale_seconds", 1.0))
        self.lock = threading.Lock()
        self.class_name = None
        self.prediction_time = None
        self.bridge = CvBridge()
        self.grid_subscriber = LatestBevGridSubscriber(
            topics["bev_points_topic"],
            grid_config["forward_range_m"],
            grid_config["map_width_m"],
            grid_config["forward_bins"],
            grid_config["lateral_bins"],
        )
        self.publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/corridor_classifier/bev_overlay"),
            RosImage,
            queue_size=1,
        )
        rospy.Subscriber(
            topics["passage_type_topic"],
            cmd_dir_intersection,
            self._passage_callback,
            queue_size=1,
        )

    def _passage_callback(self, message):
        with self.lock:
            self.class_name = message.intersection_name
            self.prediction_time = rospy.get_time()

    def publish(self):
        grid, _ = self.grid_subscriber.latest()
        with self.lock:
            class_name = self.class_name
            prediction_time = self.prediction_time
        stale = (
            prediction_time is None
            or rospy.get_time() - prediction_time > self.stale_seconds
        )
        image = render_passage_bev(
            grid,
            self.forward_range_m,
            self.map_width_m,
            class_name,
            stale=stale,
            min_points=self.min_points,
            binary=self.binary,
        )
        self.publisher.publish(
            self.bridge.cv2_to_imgmsg(np.asarray(image), encoding="rgb8")
        )


def main():
    rospy.init_node("corridor_bev_overlay")
    config = load_config(rospy.get_param("~config_dir", None))
    node = PassageBevOverlay(config)
    rate = rospy.Rate(float(rospy.get_param("~rate", 4.0)))
    while not rospy.is_shutdown():
        node.publish()
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
