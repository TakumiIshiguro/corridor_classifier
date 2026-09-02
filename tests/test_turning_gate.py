import pytest

from corridor_classifier.turning_gate import is_turning


@pytest.mark.parametrize(
    "cmd_dir,angular_z,threshold,expected",
    [
        # Straight cmd_dir never counts as turning, regardless of angular_z
        # (e.g. vnm_ros/CARE steering while scenario_navigation still
        # commands straight).
        ([1, 0, 0], 5.0, 0.12, False),
        ([1, 0, 0], 0.0, 0.12, False),
        # Non-straight cmd_dir alone is not enough either: stop ([0,0,0])
        # is "not straight" but the robot is not turning.
        ([0, 0, 0], 0.0, 0.12, False),
        ([0, 0, 0], 0.05, 0.12, False),
        # Both conditions must hold.
        ([0, 1, 0], 0.05, 0.12, False),
        ([0, 1, 0], 0.12, 0.12, True),
        ([0, 0, 1], -0.2, 0.12, True),
    ],
)
def test_is_turning_requires_nonstraight_cmd_dir_and_angular_speed(
    cmd_dir, angular_z, threshold, expected
):
    assert is_turning(cmd_dir, angular_z, threshold) is expected
