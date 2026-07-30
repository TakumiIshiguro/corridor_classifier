from PIL import Image

from corridor_classifier.feature_visualization import (
    make_feature_panel,
    representative_indices,
)


def test_representative_indices_prioritize_different_classes():
    labels = [0] * 10 + [1] * 10 + [2] * 10
    first = representative_indices(
        labels,
        max_images=6,
        min_images_per_class=2,
        seed=7,
    )
    repeated = representative_indices(
        labels,
        max_images=6,
        min_images_per_class=2,
        seed=7,
    )
    second = representative_indices(
        labels,
        max_images=6,
        min_images_per_class=2,
        seed=8,
    )
    selected_labels = [labels[index] for index in first]

    assert selected_labels.count(0) == 2
    assert selected_labels.count(1) == 2
    assert selected_labels.count(2) == 2
    assert first == repeated
    assert first != second


def test_feature_panel_places_input_and_features_side_by_side(tmp_path):
    panel = make_feature_panel(
        source_image=Image.new("RGB", (640, 480)),
        feature_map=Image.new("RGB", (224, 224)),
        class_name="straight_road",
        confidence=0.8,
        probabilities=[0.8, 0.2],
    )

    assert panel.mode == "RGB"
    assert panel.size == (448, 256)

    output_path = tmp_path / "feature_panel.png"
    panel.save(output_path, format="PNG")
    with Image.open(output_path) as saved:
        assert saved.format == "PNG"
        assert saved.size == (448, 256)


def test_feature_panel_adds_imagenet_comparison_column():
    panel = make_feature_panel(
        source_image=Image.new("RGB", (640, 480)),
        feature_map=Image.new("RGB", (224, 224)),
        class_name="straight_road",
        confidence=0.8,
        probabilities=[0.8, 0.2],
        comparison_feature_map=Image.new("RGB", (224, 224)),
        comparison_name="ImageNet ViT-S/16",
    )

    assert panel.mode == "RGB"
    assert panel.size == (672, 256)


def test_feature_panel_adds_resnet_comparison_column():
    panel = make_feature_panel(
        source_image=Image.new("RGB", (640, 480)),
        feature_map=Image.new("RGB", (224, 224)),
        class_name="straight_road",
        confidence=0.8,
        probabilities=[0.8, 0.2],
        comparison_feature_map=Image.new("RGB", (224, 224)),
        comparison_name="ImageNet ViT-S/16",
        resnet_feature_map=Image.new("RGB", (224, 224)),
        resnet_name="ImageNet ResNet-18",
    )

    assert panel.mode == "RGB"
    assert panel.size == (896, 256)
