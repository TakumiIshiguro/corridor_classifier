from threading import Lock
from typing import Optional, Sequence

import rospy
from geometry_msgs.msg import Twist
from scenario_navigation_msgs.msg import cmd_dir_intersection

STRAIGHT_CMD_DIR = (1, 0, 0)


def is_turning(
    cmd_dir: Sequence[int],
    angular_z: float,
    threshold_rad_s: float,
) -> bool:
    """Both signals must agree: scenario_navigation must be commanding
    something other than straight ([1, 0, 0]), and the robot must actually
    be rotating at or above the threshold. cmd_dir alone is not enough --
    stop ([0, 0, 0]) is also "not straight" but is not turning, and cmd_vel
    alone is not enough -- vnm_ros/CARE's own obstacle-avoidance steering
    can produce a large angular velocity while scenario_navigation is still
    commanding straight, which requiring cmd_dir here rules out without a
    separate CARE-avoidance signal.
    """
    return (
        tuple(int(v) for v in cmd_dir) != STRAIGHT_CMD_DIR
        and abs(float(angular_z)) >= float(threshold_rad_s)
    )


class CmdDirTurningGate:
    """Watches scenario_navigation's commanded-direction topic (e.g.
    /cmd_dir_intersection, scenario_navigation_msgs/cmd_dir_intersection)
    together with a Twist command topic (e.g. /cmd_vel), and reports
    turning only while both agree (see is_turning() above). Used to skip
    classification entirely while turning, since the model is not trained
    to predict passage shape mid-turn (see README.md).
    """

    def __init__(
        self,
        cmd_dir_topic: str,
        cmd_vel_topic: str,
        threshold_rad_s: float,
        stale_timeout_seconds: float = 1.0,
    ):
        self.threshold_rad_s = float(threshold_rad_s)
        self.stale_timeout_seconds = float(stale_timeout_seconds)
        self._lock = Lock()
        self._cmd_dir: Optional[tuple] = None
        self._cmd_dir_received_at: Optional[float] = None
        self._angular_z = 0.0
        self._cmd_vel_received_at: Optional[float] = None
        self._cmd_dir_subscriber = rospy.Subscriber(
            cmd_dir_topic, cmd_dir_intersection, self._cmd_dir_callback, queue_size=1
        )
        self._cmd_vel_subscriber = rospy.Subscriber(
            cmd_vel_topic, Twist, self._cmd_vel_callback, queue_size=1
        )

    def _cmd_dir_callback(self, msg: cmd_dir_intersection) -> None:
        with self._lock:
            self._cmd_dir = tuple(int(v) for v in msg.cmd_dir)
            self._cmd_dir_received_at = rospy.get_time()

    def _cmd_vel_callback(self, msg: Twist) -> None:
        with self._lock:
            self._angular_z = float(msg.angular.z)
            self._cmd_vel_received_at = rospy.get_time()

    def is_turning(self) -> bool:
        with self._lock:
            cmd_dir = self._cmd_dir
            cmd_dir_received_at = self._cmd_dir_received_at
            angular_z = self._angular_z
            cmd_vel_received_at = self._cmd_vel_received_at
        now = rospy.get_time()
        # No recent reading on either topic: treat as not turning rather
        # than guessing, since a stale reading could otherwise wedge the
        # node into permanently skipping inference.
        if cmd_dir_received_at is None or now - cmd_dir_received_at > self.stale_timeout_seconds:
            return False
        if cmd_vel_received_at is None or now - cmd_vel_received_at > self.stale_timeout_seconds:
            return False
        return is_turning(cmd_dir, angular_z, self.threshold_rad_s)
