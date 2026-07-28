import random
from typing import Sequence

from PIL import Image, ImageDraw


def representative_indices(
    class_indices: Sequence[int],
    max_images: int,
    min_images_per_class: int = 1,
    seed: int = 0,
):
    if not class_indices:
        raise ValueError("class_indices must not be empty")
    max_images = int(max_images)
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    min_images_per_class = int(min_images_per_class)
    if min_images_per_class <= 0:
        raise ValueError("min_images_per_class must be positive")

    by_class = {}
    for sample_index, class_index in enumerate(class_indices):
        by_class.setdefault(int(class_index), []).append(sample_index)
    insufficient = {
        class_index: len(indices)
        for class_index, indices in by_class.items()
        if len(indices) < min_images_per_class
    }
    if insufficient:
        raise ValueError(
            "classes do not contain enough samples for visualization: "
            f"{insufficient}, required={min_images_per_class}"
        )
    required_count = len(by_class) * min_images_per_class
    if max_images < required_count:
        raise ValueError(
            f"max_images={max_images} is smaller than the required "
            f"{required_count} images for {len(by_class)} classes"
        )

    generator = random.Random(int(seed))
    selected = set()
    for _, indices in sorted(by_class.items()):
        selected.update(
            generator.sample(indices, min_images_per_class)
        )
    remaining = [
        sample_index
        for sample_index in range(len(class_indices))
        if sample_index not in selected
    ]
    generator.shuffle(remaining)
    selected.update(remaining[: max_images - len(selected)])
    return sorted(selected)


def make_feature_panel(
    source_image: Image.Image,
    feature_map: Image.Image,
    class_name: str,
    confidence: float,
    probabilities: Sequence[float],
    label_name: str = None,
) -> Image.Image:
    image_size = (224, 224)
    header_height = 32
    source = source_image.convert("RGB").resize(
        image_size,
        Image.Resampling.BICUBIC,
    )
    features = feature_map.convert("RGB").resize(
        image_size,
        Image.Resampling.NEAREST,
    )
    panel = Image.new(
        "RGB",
        (image_size[0] * 2, image_size[1] + header_height),
        color=(20, 20, 20),
    )
    panel.paste(source, (0, header_height))
    panel.paste(features, (image_size[0], header_height))
    draw = ImageDraw.Draw(panel)
    result_text = f"pred={class_name} ({float(confidence):.3f})"
    if label_name is not None:
        result_text = f"label={label_name}  {result_text}"
    draw.text(
        (6, 5),
        f"input | DINOv2 patch features    {result_text}",
        fill=(255, 255, 255),
    )
    top_classes = sorted(
        enumerate(probabilities),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:3]
    draw.text(
        (6, 18),
        "top3: "
        + ", ".join(
            f"{index}={float(probability):.2f}"
            for index, probability in top_classes
        ),
        fill=(190, 190, 190),
    )
    return panel
