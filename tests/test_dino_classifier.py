import torch
from PIL import Image
from torch import nn

from corridor_classifier.dino_classifier import (
    DINOClassifier,
    PretrainedCNNFeatureExtractor,
    PretrainedViTFeatureExtractor,
    build_transform,
    create_dino_model,
    extract_state_dict,
    visualize_patch_features,
    visualize_spatial_features,
)


CLASS_NAMES = (
    "straight_road",
    "dead_end",
    "corner_right",
    "corner_left",
    "cross_road",
    "3_way_right",
    "3_way_center",
    "3_way_left",
)


class TinyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(3, num_classes)

    def forward(self, value):
        return self.head(self.pool(value).flatten(1))


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = type("PatchEmbed", (), {"grid_size": (2, 2)})()
        self.num_prefix_tokens = 1

    def forward_features(self, value):
        features = torch.arange(
            5 * 8,
            dtype=value.dtype,
            device=value.device,
        ).reshape(1, 5, 8)
        return features.repeat(value.shape[0], 1, 1)


class TinyCNN(nn.Module):
    def forward_features(self, value):
        features = torch.arange(
            8 * 2 * 2,
            dtype=value.dtype,
            device=value.device,
        ).reshape(1, 8, 2, 2)
        return features.repeat(value.shape[0], 1, 1, 1)


def test_transform_produces_expected_shape():
    transform = build_transform([224, 224])
    tensor = transform(Image.new("RGB", (640, 480), color=(20, 40, 60)))

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_patch_features_are_visualized_as_rgb_image():
    features = torch.randn(1, 17, 32)
    feature_map = visualize_patch_features(
        features,
        grid_size=(4, 4),
        num_prefix_tokens=1,
        output_size=(224, 224),
    )
    repeated = visualize_patch_features(
        features,
        grid_size=(4, 4),
        num_prefix_tokens=1,
        output_size=(224, 224),
    )

    assert feature_map.mode == "RGB"
    assert feature_map.size == (224, 224)
    assert feature_map.tobytes() == repeated.tobytes()


def test_spatial_features_are_visualized_as_rgb_image():
    features = torch.randn(1, 32, 4, 4)
    feature_map = visualize_spatial_features(
        features,
        output_size=(224, 224),
    )

    assert feature_map.mode == "RGB"
    assert feature_map.size == (224, 224)


def test_extract_state_dict_strips_distributed_prefix():
    checkpoint = {
        "model_state_dict": {
            "module.head.weight": torch.zeros(8, 3),
            "module.head.bias": torch.zeros(8),
        }
    }
    state_dict = extract_state_dict(checkpoint)

    assert set(state_dict) == {"head.weight", "head.bias"}


def test_classifier_loads_checkpoint_and_predicts(monkeypatch, tmp_path):
    def fake_create_model(
        model_name,
        pretrained,
        img_size,
        num_classes,
    ):
        del model_name, pretrained, img_size
        return TinyClassifier(num_classes)

    monkeypatch.setattr(
        "corridor_classifier.dino_classifier.timm.create_model",
        fake_create_model,
    )
    model = TinyClassifier(len(CLASS_NAMES))
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[4] = 5.0

    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
        },
        checkpoint_path,
    )
    config = {
        "model_name": "fake_dino",
        "input_size": [32, 32],
        "num_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES),
        "device": "cpu",
        "use_fp16": True,
        "strict_checkpoint": True,
    }

    classifier = DINOClassifier(config, str(checkpoint_path))
    prediction = classifier.predict(Image.new("RGB", (64, 48)))

    assert prediction.class_index == 4
    assert prediction.confidence > 0.9
    assert len(prediction.probabilities) == 8
    assert abs(sum(prediction.probabilities) - 1.0) < 1e-6


def test_local_pretrained_weights_are_passed_to_timm(monkeypatch, tmp_path):
    calls = {}

    def fake_create_model(model_name, **kwargs):
        calls["model_name"] = model_name
        calls.update(kwargs)
        return TinyClassifier(kwargs["num_classes"])

    monkeypatch.setattr(
        "corridor_classifier.dino_classifier.timm.create_model",
        fake_create_model,
    )
    weights_path = tmp_path / "dinov2.pth"
    weights_path.write_bytes(b"placeholder")

    create_dino_model(
        model_name="fake_dino",
        input_size=[224, 224],
        num_classes=8,
        pretrained=True,
        pretrained_weights_path=str(weights_path),
    )

    assert calls["pretrained"] is True
    assert calls["img_size"] == 224
    assert calls["pretrained_cfg_overlay"] == {
        "file": str(weights_path)
    }


def test_pretrained_vit_extracts_rgb_patch_features(monkeypatch):
    calls = {}

    def fake_create_model(model_name, **kwargs):
        calls["model_name"] = model_name
        calls.update(kwargs)
        return TinyViT()

    monkeypatch.setattr(
        "corridor_classifier.dino_classifier.timm.create_model",
        fake_create_model,
    )
    extractor = PretrainedViTFeatureExtractor(
        model_name="fake_imagenet_vit",
        input_size=[32, 32],
        device_name="cpu",
        use_fp16=True,
    )
    feature_map = extractor.extract(Image.new("RGB", (64, 48)))

    assert calls["model_name"] == "fake_imagenet_vit"
    assert calls["pretrained"] is True
    assert calls["num_classes"] == 0
    assert feature_map.mode == "RGB"
    assert feature_map.size == (32, 32)


def test_pretrained_cnn_extracts_rgb_spatial_features(monkeypatch):
    calls = {}

    def fake_create_model(model_name, **kwargs):
        calls["model_name"] = model_name
        calls.update(kwargs)
        return TinyCNN()

    monkeypatch.setattr(
        "corridor_classifier.dino_classifier.timm.create_model",
        fake_create_model,
    )
    extractor = PretrainedCNNFeatureExtractor(
        model_name="fake_imagenet_resnet",
        input_size=[32, 32],
        device_name="cpu",
        use_fp16=True,
    )
    feature_map = extractor.extract(Image.new("RGB", (64, 48)))

    assert calls["model_name"] == "fake_imagenet_resnet"
    assert calls["pretrained"] is True
    assert calls["num_classes"] == 0
    assert "img_size" not in calls
    assert feature_map.mode == "RGB"
    assert feature_map.size == (32, 32)
