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
    feature_name: str = "DINOv2 fine-tuned",
    comparison_feature_map: Image.Image = None,
    comparison_name: str = "ImageNet ViT",
    resnet_feature_map: Image.Image = None,
    resnet_name: str = "ImageNet ResNet",
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
    columns = [
        ("input", source),
        (str(feature_name), features),
    ]
    if comparison_feature_map is not None:
        comparison = comparison_feature_map.convert("RGB").resize(
            image_size,
            Image.Resampling.NEAREST,
        )
        columns.append((str(comparison_name), comparison))
    if resnet_feature_map is not None:
        resnet = resnet_feature_map.convert("RGB").resize(
            image_size,
            Image.Resampling.NEAREST,
        )
        columns.append((str(resnet_name), resnet))
    panel = Image.new(
        "RGB",
        (image_size[0] * len(columns), image_size[1] + header_height),
        color=(20, 20, 20),
    )
    for column_index, (_, image) in enumerate(columns):
        panel.paste(
            image,
            (column_index * image_size[0], header_height),
        )
    draw = ImageDraw.Draw(panel)
    for column_index, (name, _) in enumerate(columns):
        draw.text(
            (column_index * image_size[0] + 6, 5),
            name,
            fill=(255, 255, 255),
        )
    result_text = f"pred={class_name} ({float(confidence):.3f})"
    if label_name is not None:
        result_text = f"label={label_name}  {result_text}"
    draw.text(
        (6, 18),
        result_text,
        fill=(190, 190, 190),
    )
    top_classes = sorted(
        enumerate(probabilities),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:3]
    draw.text(
        (image_size[0] + 6, 18),
        "top3: "
        + ", ".join(
            f"{index}={float(probability):.2f}"
            for index, probability in top_classes
        ),
        fill=(190, 190, 190),
    )
    return panel
