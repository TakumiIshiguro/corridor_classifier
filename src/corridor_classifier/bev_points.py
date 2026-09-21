"""Turn the BEV point cloud from unidepth_ros into an occupancy grid.

unidepth_ros publishes `bev_points_topic` as robot-frame points already
cropped to the BEV band (see its `bev_projection` config). Binning here
mirrors scripts/add_bev_grid_to_dataset.py's bev_occupancy_grid so the
runtime grid matches the one the model was trained on: channel 0 is the
point count per cell, channel 1 the mean height of those points. The
published cloud is flat (z=0), which makes channel 1 all zeros -- that is
what `bev_drop_height` is for, and the production config sets it.
"""
import threading
from typing import Optional, Sequence, Tuple

import numpy as np


def bev_occupancy_grid(
    points: np.ndarray,
    forward_range_m: Tuple[float, float],
    map_width_m: float,
    forward_bins: int,
    lateral_bins: int,
) -> np.ndarray:
    minimum_forward_m, maximum_forward_m = forward_range_m
    grid = np.zeros((int(forward_bins), int(lateral_bins), 2), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return grid

    forward, left, height = points[:, 0], points[:, 1], points[:, 2]
    lateral_limit = float(map_width_m) / 2.0
    keep = (
        (forward >= minimum_forward_m)
        & (forward < maximum_forward_m)
        & (left >= -lateral_limit)
        & (left < lateral_limit)
    )
    if not keep.any():
        return grid

    forward_span = float(maximum_forward_m) - float(minimum_forward_m)
    forward_index = (
        (forward[keep] - minimum_forward_m) / forward_span * int(forward_bins)
    ).astype(np.int32).clip(0, int(forward_bins) - 1)
    lateral_index = (
        (left[keep] + lateral_limit) / (2.0 * lateral_limit) * int(lateral_bins)
    ).astype(np.int32).clip(0, int(lateral_bins) - 1)

    np.add.at(grid[..., 0], (forward_index, lateral_index), 1.0)
    np.add.at(grid[..., 1], (forward_index, lateral_index), height[keep])
    occupied = grid[..., 0] > 0
    grid[..., 1][occupied] /= grid[..., 0][occupied]
    return grid


class LatestBevGridSubscriber:
    """Keeps the most recent BEV grid, binned as messages arrive."""

    def __init__(
        self,
        topic: str,
        forward_range_m: Sequence[float],
        map_width_m: float,
        forward_bins: int,
        lateral_bins: int,
    ):
        import rospy
        from sensor_msgs.msg import PointCloud2

        self.forward_range_m = (
            float(forward_range_m[0]),
            float(forward_range_m[1]),
        )
        self.map_width_m = float(map_width_m)
        self.forward_bins = int(forward_bins)
        self.lateral_bins = int(lateral_bins)
        self._lock = threading.Lock()
        self._grid = None
        self._stamp = None
        self._subscriber = rospy.Subscriber(
            topic, PointCloud2, self._callback, queue_size=1
        )

    def _callback(self, message) -> None:
        from sensor_msgs import point_cloud2

        points = np.array(
            list(
                point_cloud2.read_points(
                    message, field_names=("x", "y", "z"), skip_nans=True
                )
            ),
            dtype=np.float32,
        )
        grid = bev_occupancy_grid(
            points,
            self.forward_range_m,
            self.map_width_m,
            self.forward_bins,
            self.lateral_bins,
        )
        with self._lock:
            self._grid = grid
            self._stamp = message.header.stamp.to_sec()

    def latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._grid is None:
                return None, None
            return self._grid.copy(), self._stamp
