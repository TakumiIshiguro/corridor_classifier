from threading import Lock
from typing import Optional

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


def is_turning(angular_z: float, threshold_rad_s: float) -> bool:
    return abs(float(angular_z)) >= float(threshold_rad_s)


def combine_turning_signals(cmd_vel_turning: bool, care_avoidance_active: bool) -> bool:
    """A CARE obstacle-avoidance correction (or in-place Safe-FOV recovery
    spin) can command a large angular velocity that looks like a scenario
    turn to a pure cmd_vel threshold, even though the robot has not actually
    reached a new corridor segment. Never report turning while CARE reports
    it is actively adjusting the trajectory for obstacles.
    """
    return bool(cmd_vel_turning) and not bool(care_avoidance_active)


class CmdVelTurningGate:
    """Watches a Twist command topic (e.g. /cmd_vel) and reports whether the
    robot is currently commanded to turn faster than a threshold angular
    speed. Used to skip classification entirely while turning, since the
    model is not trained to predict passage shape during a turn (see
    README.md) -- an existing, more direct signal (the commanded angular
    velocity) is used instead of a vision-based guess.

    Optionally also watches a CARE avoidance-active topic (std_msgs/Bool,
    e.g. /vnm/care_avoidance_active) so that CARE's own obstacle-avoidance
    steering -- which can produce a large angular velocity unrelated to any
    real intersection turn -- is never reported as turning. Disabled by
    default (empty topic name), since corridor_classifier does not require
    vnm_ros/CARE to be running.
    """

    def __init__(
        self,
        topic: str,
        threshold_rad_s: float,
        stale_timeout_seconds: float = 1.0,
        care_avoidance_topic: str = "",
    ):
        self.threshold_rad_s = float(threshold_rad_s)
        self.stale_timeout_seconds = float(stale_timeout_seconds)
        self._lock = Lock()
        self._angular_z = 0.0
        self._received_at: Optional[float] = None
        self._subscriber = rospy.Subscriber(topic, Twist, self._callback, queue_size=1)

        self._care_avoidance_active = False
        self._care_received_at: Optional[float] = None
        self._care_subscriber = None
        care_avoidance_topic = str(care_avoidance_topic).strip()
        if care_avoidance_topic:
            self._care_subscriber = rospy.Subscriber(
                care_avoidance_topic, Bool, self._care_callback, queue_size=1
            )

    def _callback(self, msg: Twist) -> None:
        with self._lock:
            self._angular_z = float(msg.angular.z)
            self._received_at = rospy.get_time()

    def _care_callback(self, msg: Bool) -> None:
        with self._lock:
            self._care_avoidance_active = bool(msg.data)
            self._care_received_at = rospy.get_time()

    def _care_avoidance_currently_active(self) -> bool:
        if self._care_subscriber is None:
            return False
        with self._lock:
            active = self._care_avoidance_active
            received_at = self._care_received_at
        if received_at is None:
            return False
        if rospy.get_time() - received_at > self.stale_timeout_seconds:
            # No recent CARE state: do not let a stale "avoidance active"
            # reading permanently suppress real turning detection.
            return False
        return active

    def is_turning(self) -> bool:
        with self._lock:
            angular_z = self._angular_z
            received_at = self._received_at
        if received_at is None:
            return False
        if rospy.get_time() - received_at > self.stale_timeout_seconds:
            # No recent cmd_vel: treat as not turning rather than guessing,
            # since a stale reading could otherwise wedge the node into
            # permanently skipping inference.
            return False
        cmd_vel_turning = is_turning(angular_z, self.threshold_rad_s)
        return combine_turning_signals(
            cmd_vel_turning, self._care_avoidance_currently_active()
        )
