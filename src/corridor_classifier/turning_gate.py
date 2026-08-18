from threading import Lock
from typing import Optional

import rospy
from geometry_msgs.msg import Twist


def is_turning(angular_z: float, threshold_rad_s: float) -> bool:
    return abs(float(angular_z)) >= float(threshold_rad_s)


class CmdVelTurningGate:
    """Watches a Twist command topic (e.g. /cmd_vel) and reports whether the
    robot is currently commanded to turn faster than a threshold angular
    speed. Used to skip classification entirely while turning, since the
    model is not trained to predict passage shape during a turn (see
    README.md) -- an existing, more direct signal (the commanded angular
    velocity) is used instead of a vision-based guess.
    """

    def __init__(self, topic: str, threshold_rad_s: float, stale_timeout_seconds: float = 1.0):
        self.threshold_rad_s = float(threshold_rad_s)
        self.stale_timeout_seconds = float(stale_timeout_seconds)
        self._lock = Lock()
        self._angular_z = 0.0
        self._received_at: Optional[float] = None
        self._subscriber = rospy.Subscriber(topic, Twist, self._callback, queue_size=1)

    def _callback(self, msg: Twist) -> None:
        with self._lock:
            self._angular_z = float(msg.angular.z)
            self._received_at = rospy.get_time()

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
        return is_turning(angular_z, self.threshold_rad_s)
