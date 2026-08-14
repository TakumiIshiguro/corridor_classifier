from threading import Lock

import message_filters
from sensor_msgs.msg import Image


class LatestRgbDepthSubscriber:
    def __init__(self, image_topic: str, depth_topic: str):
        self._lock = Lock()
        self._latest = None
        self._image = message_filters.Subscriber(image_topic, Image)
        self._depth = message_filters.Subscriber(depth_topic, Image)
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._image, self._depth],
            queue_size=5,
            slop=0.05,
        )
        self._synchronizer.registerCallback(self._callback)

    def _callback(self, image_message: Image, depth_message: Image) -> None:
        with self._lock:
            self._latest = (image_message, depth_message)

    def take_latest(self):
        with self._lock:
            messages = self._latest
            self._latest = None
        return messages
