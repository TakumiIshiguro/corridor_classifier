from bisect import bisect_left, bisect_right
from math import atan2, cos, sin
from statistics import median
from typing import Iterable, List, Sequence, Tuple


def quaternion_yaw(orientation) -> float:
    return atan2(
        2.0
        * (
            float(orientation.w) * float(orientation.z)
            + float(orientation.x) * float(orientation.y)
        ),
        1.0
        - 2.0
        * (
            float(orientation.y) * float(orientation.y)
            + float(orientation.z) * float(orientation.z)
        ),
    )


def _unwrap_yaws(yaws: Sequence[float]) -> List[float]:
    if not yaws:
        return []
    result = [float(yaws[0])]
    previous = float(yaws[0])
    for yaw in yaws[1:]:
        yaw = float(yaw)
        delta = atan2(sin(yaw - previous), cos(yaw - previous))
        result.append(result[-1] + delta)
        previous = yaw
    return result


def _interpolate(stamps, values, stamp: float) -> float:
    index = bisect_right(stamps, stamp)
    if index <= 0:
        return float(values[0])
    if index >= len(stamps):
        return float(values[-1])
    left_stamp = float(stamps[index - 1])
    right_stamp = float(stamps[index])
    if right_stamp <= left_stamp:
        return float(values[index])
    ratio = (float(stamp) - left_stamp) / (right_stamp - left_stamp)
    return float(values[index - 1]) + ratio * (
        float(values[index]) - float(values[index - 1])
    )


class TurningDetector:
    """Detect sustained turns from smoothed localization yaw changes."""

    def __init__(
        self,
        stamps: Sequence[float],
        yaws: Sequence[float],
        angular_speed_threshold_rad_s: float,
        window_seconds: float,
        minimum_duration_seconds: float = 0.5,
        padding_seconds: float = 0.25,
        maximum_pose_gap_seconds: float = 0.5,
    ):
        if len(stamps) != len(yaws) or len(stamps) < 2:
            raise ValueError("turn detection requires at least two stamped poses")
        if angular_speed_threshold_rad_s <= 0.0:
            raise ValueError("angular speed threshold must be positive")
        if window_seconds <= 0.0:
            raise ValueError("turn detection window must be positive")
        if minimum_duration_seconds < 0.0 or padding_seconds < 0.0:
            raise ValueError("turn duration and padding must be non-negative")
        if maximum_pose_gap_seconds <= 0.0:
            raise ValueError("maximum pose gap must be positive")

        ordered = sorted(
            (float(stamp), float(yaw))
            for stamp, yaw in zip(stamps, yaws)
        )
        self.stamps = [item[0] for item in ordered]
        self.yaws = _unwrap_yaws([item[1] for item in ordered])
        self.threshold = float(angular_speed_threshold_rad_s)
        self.window_seconds = float(window_seconds)
        self.minimum_duration_seconds = float(minimum_duration_seconds)
        self.padding_seconds = float(padding_seconds)
        self.maximum_pose_gap_seconds = float(maximum_pose_gap_seconds)
        self.intervals = self._build_intervals()
        self._interval_starts = [interval[0] for interval in self.intervals]

    def angular_speed(self, stamp: float) -> float:
        half_window = 0.5 * self.window_seconds
        start = float(stamp) - half_window
        end = float(stamp) + half_window
        if start < self.stamps[0] or end > self.stamps[-1]:
            return 0.0
        start_index = bisect_right(self.stamps, start)
        end_index = bisect_left(self.stamps, end)
        if start_index >= len(self.stamps) or end_index <= 0:
            return 0.0
        if self.stamps[start_index] - self.stamps[start_index - 1] > (
            self.maximum_pose_gap_seconds
        ):
            return 0.0
        if end_index < len(self.stamps) and self.stamps[end_index] - self.stamps[
            end_index - 1
        ] > self.maximum_pose_gap_seconds:
            return 0.0
        yaw_change = abs(
            _interpolate(self.stamps, self.yaws, end)
            - _interpolate(self.stamps, self.yaws, start)
        )
        return yaw_change / self.window_seconds

    def _build_intervals(self) -> List[Tuple[float, float]]:
        candidates = [
            stamp
            for stamp in self.stamps
            if self.angular_speed(stamp) >= self.threshold
        ]
        if not candidates:
            return []
        pose_deltas = [
            later - earlier
            for earlier, later in zip(self.stamps, self.stamps[1:])
            if 0.0 < later - earlier <= self.maximum_pose_gap_seconds
        ]
        typical_delta = median(pose_deltas) if pose_deltas else 0.0
        groups = []
        start = previous = candidates[0]
        for stamp in candidates[1:]:
            if stamp - previous > self.maximum_pose_gap_seconds:
                groups.append((start, previous))
                start = stamp
            previous = stamp
        groups.append((start, previous))

        intervals = []
        for start, end in groups:
            duration = end - start + typical_delta
            if duration + 1.0e-9 < self.minimum_duration_seconds:
                continue
            intervals.append(
                (
                    start - self.padding_seconds,
                    end + self.padding_seconds,
                )
            )
        return intervals

    def is_turning(self, stamp: float) -> bool:
        if not self.intervals:
            return False
        index = bisect_right(self._interval_starts, float(stamp)) - 1
        return index >= 0 and float(stamp) <= self.intervals[index][1]


def detector_from_pose_messages(
    messages: Iterable[Tuple[str, object, object]],
    **kwargs,
) -> TurningDetector:
    stamps = []
    yaws = []
    for _, message, bag_time in messages:
        stamps.append(float(bag_time.to_sec()))
        yaws.append(quaternion_yaw(message.pose.pose.orientation))
    return TurningDetector(stamps=stamps, yaws=yaws, **kwargs)


def _label_runs(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    if not labels:
        return []
    runs = []
    start = 0
    current = int(labels[0])
    for index, label in enumerate(labels[1:], start=1):
        label = int(label)
        if label == current:
            continue
        runs.append((current, start, index - 1))
        current = label
        start = index
    runs.append((current, start, len(labels) - 1))
    return runs


def replace_post_turn_label_islands(
    stamps: Sequence[float],
    labels: Sequence[int],
    turn_class_index: int,
    maximum_seconds: float,
) -> List[int]:
    """Replace a short A -> turning -> A -> B island with B labels."""
    if len(stamps) != len(labels):
        raise ValueError("post-turn stamps and labels must have equal length")
    if maximum_seconds < 0.0:
        raise ValueError("post-turn maximum duration must be non-negative")
    refined = [int(label) for label in labels]
    if not refined or maximum_seconds == 0.0:
        return refined

    stamps = [float(stamp) for stamp in stamps]
    turn_class_index = int(turn_class_index)
    runs = _label_runs(refined)
    for index in range(1, len(runs) - 2):
        before, turning, stale, following = runs[index - 1 : index + 3]
        if turning[0] != turn_class_index:
            continue
        if stale[0] != before[0]:
            continue
        if following[0] in (stale[0], turn_class_index):
            continue
        duration = stamps[stale[2]] - stamps[stale[1]]
        if duration < 0.0 or duration > maximum_seconds:
            continue
        refined[stale[1] : stale[2] + 1] = [following[0]] * (
            stale[2] - stale[1] + 1
        )
    return refined


def bridge_short_turning_gaps(
    stamps: Sequence[float],
    labels: Sequence[int],
    turn_class_index: int,
    maximum_seconds: float,
) -> List[int]:
    """Replace a short turning -> A -> turning gap with turning labels."""
    if len(stamps) != len(labels):
        raise ValueError("turning-gap stamps and labels must have equal length")
    if maximum_seconds < 0.0:
        raise ValueError("turning-gap maximum duration must be non-negative")
    refined = [int(label) for label in labels]
    if not refined or maximum_seconds == 0.0:
        return refined

    stamps = [float(stamp) for stamp in stamps]
    turn_class_index = int(turn_class_index)
    runs = _label_runs(refined)
    for index in range(1, len(runs) - 1):
        before, gap, after = runs[index - 1 : index + 2]
        if before[0] != turn_class_index or after[0] != turn_class_index:
            continue
        if gap[0] == turn_class_index:
            continue
        duration = stamps[gap[2]] - stamps[gap[1]]
        if duration < 0.0 or duration > maximum_seconds:
            continue
        refined[gap[1] : gap[2] + 1] = [turn_class_index] * (
            gap[2] - gap[1] + 1
        )
    return refined
