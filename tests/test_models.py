from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from corridor_classifier.models import (
    CorridorPredictor,
    MultimodalCorridorModel,
    PassagePrediction,
    RGBModel,
    depth_to_tensor,
    load_model_checkpoint,
    regional_patch_features,
)


class FakeDinoFeatures(nn.Module):
    num_features = 12

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.num_features)
        # 4 patch tokens arranged as a 1x4 grid (one row, four columns).
        self.patch_embed = SimpleNamespace(grid_size=(1, 4))

    def forward(self, images):
        return self.projection(images.mean(dim=(-2, -1)))

    def get_intermediate_layers(
        self,
        images,
        n,
        return_prefix_tokens=False,
        norm=False,
    ):
        base = self.forward(images)
        outputs = []
        for index in range(int(n)):
            patch_tokens = base.unsqueeze(1).repeat(1, 4, 1) + index
            # Make each of the 4 patch columns distinguishable so regional
            # pooling tests can assert that different columns are used.
            patch_tokens = patch_tokens + torch.arange(4).view(1, 4, 1)
            prefix_tokens = base.unsqueeze(1) + index
            outputs.append(
                (patch_tokens, prefix_tokens)
                if return_prefix_tokens
                else patch_tokens
            )
        return outputs


class FakeDinoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 8)

    def forward(self, images):
        return self.head(images.mean(dim=(-2, -1)))


@pytest.mark.parametrize(
    "use_depth,use_gru,sequence_length",
    [(False, True, 5), (True, False, 1), (True, True, 5)],
)
def test_multimodal_models_return_eight_logits(
    use_depth, use_gru, sequence_length
):
    model = MultimodalCorridorModel(
        dino=FakeDinoFeatures(),
        num_classes=8,
        use_depth=use_depth,
        use_gru=use_gru,
        depth_feature_dim=8,
        fusion_dim=10,
        gru_hidden_size=7,
    )
    inputs = {
        "rgb": torch.randn(2, sequence_length, 3, 32, 32),
    }
    if use_depth:
        inputs["depth"] = torch.randn(2, sequence_length, 2, 32, 32)

    assert model(inputs).shape == (2, 8)


def test_passage_direction_model_returns_separate_heads():
    model = MultimodalCorridorModel(
        dino=FakeDinoFeatures(),
        num_classes=9,
        use_depth=True,
        use_gru=True,
        depth_feature_dim=8,
        fusion_dim=10,
        gru_hidden_size=7,
        output_mode="passage_directions",
    )
    outputs = model(
        {
            "rgb": torch.randn(2, 3, 3, 32, 32),
            "depth": torch.randn(2, 3, 2, 32, 32),
        }
    )

    assert outputs["direction_logits"].shape == (2, 3)
    assert model.classifier is None


@pytest.mark.parametrize(
    "dino_readout,expected_rgb_dim",
    [
        ("last_cls", 12),
        ("last_cls_patch_mean", 24),
        ("last4_cls", 48),
        ("last4_cls_patch_mean", 60),
        ("last_cls_regional3", 48),
    ],
)
def test_dino_readout_controls_fusion_input_dimension(
    dino_readout, expected_rgb_dim
):
    model = MultimodalCorridorModel(
        dino=FakeDinoFeatures(),
        num_classes=9,
        use_depth=True,
        use_gru=True,
        depth_feature_dim=8,
        fusion_dim=10,
        gru_hidden_size=7,
        output_mode="passage_directions",
        dino_readout=dino_readout,
    )

    outputs = model(
        {
            "rgb": torch.randn(2, 3, 3, 32, 32),
            "depth": torch.randn(2, 3, 2, 32, 32),
        }
    )

    assert model.fusion[0].normalized_shape == (expected_rgb_dim + 8,)
    assert outputs["direction_logits"].shape == (2, 3)


def test_regional_readout_preserves_left_right_distinction():
    dino = FakeDinoFeatures()
    patch_tokens = torch.zeros(1, 4, FakeDinoFeatures.num_features)
    # Columns are tagged 0..3 by FakeDinoFeatures; tensor_split(3) groups the
    # 1x4 grid into bands of sizes 2, 1, 1.
    patch_tokens += torch.arange(4).view(1, 4, 1)

    regional = regional_patch_features(dino, patch_tokens, rows=1, columns=3)

    num_features = FakeDinoFeatures.num_features
    assert regional.shape == (1, 3 * num_features)
    left, center, right = regional.split(num_features, dim=-1)
    assert torch.allclose(left, torch.full_like(left, 0.5))  # mean of columns 0,1
    assert torch.allclose(center, torch.full_like(center, 2.0))  # column 2
    assert torch.allclose(right, torch.full_like(right, 3.0))  # column 3
    assert not torch.allclose(left, right)


def test_regional_readout_requires_patch_embed_grid_size():
    dino = FakeDinoFeatures()
    del dino.patch_embed

    with pytest.raises(ValueError, match="patch_embed"):
        regional_patch_features(
            dino, torch.zeros(1, 4, FakeDinoFeatures.num_features), rows=1, columns=3
        )


def test_depth_tensor_contains_log_depth_and_validity_mask():
    depth = np.asarray([[0.1, 1.0], [10.0, np.nan]], dtype=np.float32)
    tensor = depth_to_tensor(depth, 0.1, 10.0)

    assert tensor.shape == (2, 2, 2)
    assert tensor[0, 0, 0] == pytest.approx(0.0)
    assert tensor[0, 1, 0] == pytest.approx(1.0)
    assert torch.equal(tensor[1], torch.tensor([[1.0, 1.0], [1.0, 0.0]]))


def test_depth_pool_preserves_configured_spatial_grid_before_projection():
    model = MultimodalCorridorModel(
        dino=FakeDinoFeatures(),
        num_classes=9,
        use_depth=True,
        use_gru=True,
        depth_feature_dim=8,
        depth_pool_size=4,
        fusion_dim=10,
        gru_hidden_size=7,
        output_mode="passage_directions",
    )

    outputs = model(
        {
            "rgb": torch.randn(2, 3, 3, 32, 32),
            "depth": torch.randn(2, 3, 2, 32, 32),
        }
    )

    assert model.depth_encoder.projection.in_features == 128 * 4 * 4
    assert outputs["direction_logits"].shape == (2, 3)


def test_predictor_waits_for_full_strided_context(monkeypatch):
    model = MultimodalCorridorModel(
        dino=FakeDinoFeatures(),
        num_classes=8,
        use_depth=False,
        use_gru=True,
        fusion_dim=10,
        gru_hidden_size=7,
    )
    observed = {}
    original_classify = model.classify_features

    def record_sequence(features):
        observed["shape"] = tuple(features.shape)
        return original_classify(features)

    model.classify_features = record_sequence
    monkeypatch.setattr(
        "corridor_classifier.models.create_corridor_model",
        lambda config: model,
    )
    monkeypatch.setattr(
        "corridor_classifier.models.load_model_checkpoint",
        lambda model, config, path: None,
    )
    predictor = CorridorPredictor(
        {
            "architecture": "rgb_gru",
            "class_names": [str(index) for index in range(8)],
            "device": "cpu",
            "use_fp16": False,
            "input_size": [32, 32],
            "sequence_length": 3,
            "frame_stride": 4,
            "maximum_gap_seconds": 0.4,
            "use_depth": False,
        },
        "unused.pth",
    )

    outputs = [
        predictor.predict(
            Image.new("RGB", (32, 32), color=(index, index, index)),
            stamp=index * 0.25,
        )
        for index in range(9)
    ]

    assert all(output is None for output in outputs[:8])
    assert outputs[-1] is not None
    assert predictor.required_context_length == 9
    assert observed["shape"][1] == 3


def test_predictor_decodes_passage_direction_heads(monkeypatch):
    model = MultimodalCorridorModel(
        dino=FakeDinoFeatures(),
        num_classes=9,
        use_depth=False,
        use_gru=False,
        fusion_dim=10,
        output_mode="passage_directions",
    )
    with torch.no_grad():
        model.direction_classifier.weight.zero_()
        model.direction_classifier.bias.copy_(torch.tensor([10.0, -10.0, 10.0]))
    monkeypatch.setattr(
        "corridor_classifier.models.create_corridor_model",
        lambda config: model,
    )
    monkeypatch.setattr(
        "corridor_classifier.models.load_model_checkpoint",
        lambda model, config, path: None,
    )
    predictor = CorridorPredictor(
        {
            "architecture": "rgb",
            "output_mode": "passage_directions",
            "class_names": [
                "straight_road",
                "dead_end",
                "corner_right",
                "corner_left",
                "cross_road",
                "3_way_right",
                "3_way_center",
                "3_way_left",
                "turning",
            ],
            "device": "cpu",
            "use_fp16": False,
            "input_size": [32, 32],
            "sequence_length": 1,
            "frame_stride": 1,
            "maximum_gap_seconds": 0.4,
            "use_depth": False,
            "direction_thresholds": [0.5, 0.5, 0.5],
        },
        "unused.pth",
    )

    prediction = predictor.predict(Image.new("RGB", (32, 32)))

    assert isinstance(prediction, PassagePrediction)
    assert prediction.open_directions == (1, 0, 1)
    assert prediction.class_name == "3_way_right"


def test_legacy_rgb_checkpoint_is_loaded(tmp_path):
    legacy = FakeDinoClassifier()
    checkpoint = tmp_path / "legacy.pth"
    torch.save({"model_state_dict": legacy.state_dict()}, checkpoint)
    wrapped = RGBModel(FakeDinoClassifier())

    load_model_checkpoint(
        wrapped,
        {
            "architecture": "rgb",
            "class_names": [str(index) for index in range(8)],
            "strict_checkpoint": True,
        },
        str(checkpoint),
    )

    assert torch.equal(
        wrapped.state_dict()["dino.head.weight"],
        legacy.state_dict()["head.weight"],
    )
