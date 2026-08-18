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
    assert class_name_from_directions(directions) == class_name
    assert class_index_from_directions(directions, CLASS_NAMES) == (
        CLASS_NAMES.index(class_name)
    )


def test_turning_class_is_rejected():
    with pytest.raises(ValueError, match="unsupported passage class"):
        passage_target("turning")


def test_passage_counts_over_direction_only_labels():
    counts = passage_label_counts(
        [
            CLASS_NAMES.index("straight_road"),
            CLASS_NAMES.index("corner_left"),
        ],
        CLASS_NAMES,
    )

    assert counts["direction_positive"] == [1, 1, 0]
    assert counts["direction_negative"] == [1, 1, 2]
    assert counts["direction_samples"] == 2


def test_positive_weights_are_capped_and_thresholds_are_independent():
    weights = inverse_frequency_positive_weights([10, 2, 0], [10, 18, 20], 4.0)

    assert torch.equal(weights, torch.tensor([1.0, 4.0, 1.0]))
    assert threshold_directions([0.6, 0.4, 0.8], [0.5, 0.3, 0.9]) == (
        1,
        1,
        0,
    )
