import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from corridor_classifier.models import (
    CorridorPredictor,
    MultimodalCorridorModel,
    RGBModel,
    depth_to_tensor,
    load_model_checkpoint,
)


class FakeDinoFeatures(nn.Module):
    num_features = 12

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.num_features)

    def forward(self, images):
        return self.projection(images.mean(dim=(-2, -1)))


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


def test_depth_tensor_contains_log_depth_and_validity_mask():
    depth = np.asarray([[0.1, 1.0], [10.0, np.nan]], dtype=np.float32)
    tensor = depth_to_tensor(depth, 0.1, 10.0)

    assert tensor.shape == (2, 2, 2)
    assert tensor[0, 0, 0] == pytest.approx(0.0)
    assert tensor[0, 1, 0] == pytest.approx(1.0)
    assert torch.equal(tensor[1], torch.tensor([[1.0, 1.0], [1.0, 0.0]]))


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
