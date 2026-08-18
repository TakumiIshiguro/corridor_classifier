import pytest

from corridor_classifier.turning_gate import is_turning


@pytest.mark.parametrize(
    "angular_z,threshold,expected",
    [
        (0.0, 0.2, False),
        (0.19, 0.2, False),
        (0.2, 0.2, True),
        (-0.2, 0.2, True),
        (-0.5, 0.2, True),
        (0.5, 1.0, False),
    ],
)
def test_is_turning_thresholds_absolute_angular_speed(angular_z, threshold, expected):
    assert is_turning(angular_z, threshold) is expected
