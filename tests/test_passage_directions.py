import pytest
import torch

from corridor_classifier.passage_directions import (
    CLASS_TO_DIRECTIONS,
    class_index_from_directions,
    class_name_from_directions,
    inverse_frequency_positive_weights,
    passage_label_counts,
    passage_target,
    threshold_directions,
)


CLASS_NAMES = tuple(CLASS_TO_DIRECTIONS) + ("turning",)


@pytest.mark.parametrize("class_name,directions", CLASS_TO_DIRECTIONS.items())
def test_passage_classes_round_trip(class_name, directions):
    target = passage_target(class_name)

    assert tuple(target["directions"].tolist()) == directions
    assert target["direction_mask"] == 1.0
    assert target["turning"] == 0.0
    assert class_name_from_directions(directions) == class_name
    assert class_index_from_directions(directions, CLASS_NAMES) == (
        CLASS_NAMES.index(class_name)
    )


def test_turning_masks_direction_supervision():
    target = passage_target("turning")

    assert torch.equal(target["directions"], torch.zeros(3))
    assert target["direction_mask"] == 0.0
    assert target["turning"] == 1.0


def test_passage_counts_exclude_turning_from_direction_counts():
    counts = passage_label_counts(
        [CLASS_NAMES.index("straight_road"), CLASS_NAMES.index("turning")],
        CLASS_NAMES,
    )

    assert counts["direction_positive"] == [1, 0, 0]
    assert counts["direction_negative"] == [0, 1, 1]
    assert counts["direction_samples"] == 1
    assert counts["turning_positive"] == 1
    assert counts["turning_negative"] == 1


def test_positive_weights_are_capped_and_thresholds_are_independent():
    weights = inverse_frequency_positive_weights([10, 2, 0], [10, 18, 20], 4.0)

    assert torch.equal(weights, torch.tensor([1.0, 4.0, 1.0]))
    assert threshold_directions([0.6, 0.4, 0.8], [0.5, 0.3, 0.9]) == (
        1,
        1,
        0,
    )
