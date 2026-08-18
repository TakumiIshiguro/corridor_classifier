from typing import Dict, Sequence, Tuple

import torch


DIRECTION_NAMES = ("front", "left", "right")

CLASS_TO_DIRECTIONS = {
    "straight_road": (1, 0, 0),
    "dead_end": (0, 0, 0),
    "corner_right": (0, 0, 1),
    "corner_left": (0, 1, 0),
    "cross_road": (1, 1, 1),
    "3_way_right": (1, 0, 1),
    "3_way_center": (0, 1, 1),
    "3_way_left": (1, 1, 0),
}
DIRECTIONS_TO_CLASS = {
    directions: class_name
    for class_name, directions in CLASS_TO_DIRECTIONS.items()
}


def passage_target(class_name: str) -> Dict[str, torch.Tensor]:
    class_name = str(class_name)
    if class_name not in CLASS_TO_DIRECTIONS:
        raise ValueError(
            f"unsupported passage class: {class_name}. Turning frames must "
            "be filtered out of the dataset before building passage "
            "targets; this model no longer predicts turning."
        )
    return {
        "directions": torch.tensor(
            CLASS_TO_DIRECTIONS[class_name], dtype=torch.float32
        ),
    }


def passage_target_from_index(
    class_index: int,
    class_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    class_index = int(class_index)
    if class_index < 0 or class_index >= len(class_names):
        raise ValueError(
            f"class_index must be in [0, {len(class_names) - 1}]: "
            f"{class_index}"
        )
    return passage_target(class_names[class_index])


def class_name_from_directions(directions: Sequence[int]) -> str:
    values = tuple(int(bool(value)) for value in directions)
    if len(values) != 3:
        raise ValueError("directions must contain front, left, and right")
    return DIRECTIONS_TO_CLASS[values]


def class_index_from_directions(
    directions: Sequence[int],
    class_names: Sequence[str],
) -> int:
    class_name = class_name_from_directions(directions)
    try:
        return list(class_names).index(class_name)
    except ValueError as error:
        raise ValueError(
            f"class_names does not contain reconstructed class: {class_name}"
        ) from error


def passage_label_counts(
    class_indices: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, object]:
    direction_positive = torch.zeros(3, dtype=torch.int64)
    for class_index in class_indices:
        target = passage_target_from_index(class_index, class_names)
        direction_positive += target["directions"].to(torch.int64)
    direction_samples = len(class_indices)
    return {
        "direction_positive": direction_positive.tolist(),
        "direction_negative": (
            direction_samples - direction_positive
        ).tolist(),
        "direction_samples": int(direction_samples),
    }


def inverse_frequency_positive_weights(
    positive_counts: Sequence[int],
    negative_counts: Sequence[int],
    maximum_weight: float,
) -> torch.Tensor:
    positive = torch.as_tensor(positive_counts, dtype=torch.float32)
    negative = torch.as_tensor(negative_counts, dtype=torch.float32)
    if positive.shape != negative.shape or positive.numel() == 0:
        raise ValueError("positive and negative counts must have the same shape")
    weights = torch.ones_like(positive)
    available = positive > 0
    weights[available] = negative[available] / positive[available]
    return weights.clamp(min=1.0, max=float(maximum_weight))


def threshold_directions(
    probabilities: Sequence[float],
    thresholds: Sequence[float],
) -> Tuple[int, int, int]:
    if len(probabilities) != 3 or len(thresholds) != 3:
        raise ValueError("probabilities and thresholds must contain 3 values")
    return tuple(
        int(float(probability) >= float(threshold))
        for probability, threshold in zip(probabilities, thresholds)
    )
