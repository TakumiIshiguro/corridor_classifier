from threading import Lock
from typing import FrozenSet, Optional

import rospy
from std_msgs.msg import String


def parse_target_labels(data: str) -> FrozenSet[str]:
    return frozenset(label for label in str(data).split(",") if label)


class ScenarioTargetLabelsSubscriber:
    """Watches a topic (std_msgs/String, comma-separated intersection_name
    values, e.g. "3_way_center,corner_left") published by
    scenario_navigation's cmd_dir_executor describing which corridor labels
    currently satisfy the active target. While scenario_navigation is
    executing a turn step, this looks ahead to the step the turn leads
    into (see cmd_dir_executor_detailed.cpp's activeTargetLabels()).

    Used so that, while turning, a raw prediction matching that lookahead
    target can be published on /passage_type immediately instead of
    "turning" (see corridor_classifier_node.py), letting
    scenario_navigation's turnFinish() complete the step as soon as the
    destination is visible instead of always waiting for the physical turn
    (cmd_vel) to settle back down.

    Disabled by default (empty topic name): corridor_classifier does not
    require scenario_navigation to be running.
    """

    def __init__(self, topic: str, stale_timeout_seconds: float = 1.0):
        self.stale_timeout_seconds = float(stale_timeout_seconds)
        self._lock = Lock()
        self._labels: FrozenSet[str] = frozenset()
        self._received_at: Optional[float] = None
        self._subscriber = None
        topic = str(topic).strip()
        if topic:
            self._subscriber = rospy.Subscriber(topic, String, self._callback, queue_size=1)

    def _callback(self, msg: String) -> None:
        with self._lock:
            self._labels = parse_target_labels(msg.data)
            self._received_at = rospy.get_time()

    def contains(self, label: str) -> bool:
        if self._subscriber is None:
            return False
        with self._lock:
            labels = self._labels
            received_at = self._received_at
        if received_at is None:
            return False
        if rospy.get_time() - received_at > self.stale_timeout_seconds:
            # No recent scenario state: do not let a stale target set keep
            # matching a label that may no longer be relevant.
            return False
        return label in labels
