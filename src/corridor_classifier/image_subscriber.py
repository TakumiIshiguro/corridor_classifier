from threading import Lock
from typing import Optional

import rospy
from sensor_msgs.msg import Image


class LatestImageSubscriber:
    """Keep only the newest unprocessed image message."""

    def __init__(self, topic: str, queue_size: int = 1):
        self._lock = Lock()
        self._latest_msg = None
        self._received_sequence = 0
        self._consumed_sequence = 0
        self._subscriber = rospy.Subscriber(
            topic,
            Image,
            self._callback,
            queue_size=queue_size,
        )

    def _callback(self, msg: Image) -> None:
        with self._lock:
            self._latest_msg = msg
            self._received_sequence += 1

    def take_latest(self) -> Optional[Image]:
        with self._lock:
            if (
                self._latest_msg is None
                or self._received_sequence == self._consumed_sequence
            ):
                return None
            msg = self._latest_msg
            self._consumed_sequence = self._received_sequence
        return msg

