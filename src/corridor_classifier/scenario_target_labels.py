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
    currently satisfy the active scenario step's target.

    Used so a raw prediction that matches the scenario's actual target can
    bypass the direction debouncer's hold period immediately (see
    direction_debouncer.py's ``bypass_hold``), instead of the target being
    missed or delayed by up to ``confirm_frames`` while an unrelated,
    lower-priority switch is still being held.

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
            # bypassing the hold for a label that may no longer be relevant.
            return False
        return label in labels
