"""BEV data and CNN integration checks; no pretrained weights required."""

import csv
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from corridor_classifier.dataset import CorridorMultiInputDataset, load_session_samples
from corridor_classifier.models import BevEncoder, create_corridor_model


class TinyDino(nn.Module):
    num_features = 6

    def __init__(self, num_classes=0, **kwargs):
        super().__init__()
        self.projection = nn.Linear(3, self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes else nn.Identity()

    def forward(self, images):
        return self.head(self.projection(images.mean(dim=(-2, -1))))


@pytest.mark.parametrize("architecture", ["rgb_bev", "rgb_bev_gru", "rgb_depth_bev", "rgb_depth_bev_gru"])
@pytest.mark.parametrize("output_mode", ["class", "passage_directions"])
def test_bev_factory_connects_encoder_to_loss(monkeypatch, architecture, output_mode):
    monkeypatch.setattr("corridor_classifier.models.create_dino_model", TinyDino)
    torch.manual_seed(42)
    use_depth = "depth" in architecture
    use_gru = architecture.endswith("gru")
    model = create_corridor_model({
        "architecture": architecture, "model_name": "test", "input_size": [16, 16],
        "num_classes": 8, "output_mode": output_mode, "use_depth": use_depth,
        "use_gru": use_gru, "use_bev": True, "bev_feature_dim": 5,
        "bev_pool_size": 2, "depth_feature_dim": 4, "fusion_dim": 10,
        "gru_hidden_size": 7,
    })
    length = 3 if use_gru else 1
    inputs = {"rgb": torch.randn(2, length, 3, 16, 16),
              "bev": torch.rand(2, length, 2, 16, 32)}
    if use_depth:
        inputs["depth"] = torch.rand(2, length, 2, 16, 16)
    output = model(inputs)
    logits = output if output_mode == "class" else output["direction_logits"]
    assert logits.shape == (2, 8 if output_mode == "class" else 3)
    logits.square().sum().backward()
    assert getattr(model, "bev_encoder", None) is not None
    gradient = model.bev_encoder.features[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    task_ids = {id(parameter) for parameter in model.task_parameters()}
    assert all(id(parameter) in task_ids for parameter in model.bev_encoder.parameters())


def test_bev_encoder_compresses_only_counts_without_mutating_input():
    encoder = BevEncoder(5).eval()
    bev = torch.zeros(2, 2, 16, 32)
    bev[:, 0] = 99
    bev[:, 1] = 0.75
    before = bev.clone()
    observed = []
    handle = encoder.features[0].register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone())
    )
    try:
        assert encoder(bev).shape == (2, 5)
    finally:
        handle.remove()
    torch.testing.assert_close(observed[0][:, 0], torch.full((2, 16, 32), np.log(100)))
    torch.testing.assert_close(observed[0][:, 1], before[:, 1])
    torch.testing.assert_close(bev, before)


@pytest.mark.parametrize("flip", [False, True])
def test_bev_manifest_sequence_layout_and_flip(tmp_path, flip):
    rows, grids = [], []
    for index in range(3):
        Image.new("RGB", (16, 16)).save(tmp_path / f"{index}.png")
        grid = np.arange(16 * 32 * 2, dtype=np.float32).reshape(16, 32, 2) + index
        np.save(tmp_path / f"{index}.npy", grid)
        grids.append(torch.from_numpy(grid).permute(2, 0, 1))
        rows.append({"filename": f"{index}.png", "class_index": 2,
                     "stamp": index * 0.25, "bev_grid_filename": f"{index}.npy"})
    with (tmp_path / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    samples = load_session_samples(str(tmp_path), 8)
    dataset = CorridorMultiInputDataset(
        samples, [16, 16], {"sequence_length": 2, "frame_stride": 2,
                           "use_depth": False, "use_bev": True},
        {"horizontal_flip_probability": float(flip)},
    )
    inputs, label = dataset[0]
    expected = torch.stack([grids[0], grids[2]])
    if flip:
        expected = expected.flip(-1)
    torch.testing.assert_close(inputs["bev"], expected)
    assert label == (3 if flip else 2)
    samples[0] = replace(samples[0], bev_grid_path=None)
    with pytest.raises(ValueError, match="no valid sequences"):
        CorridorMultiInputDataset(samples, [16, 16], {
            "sequence_length": 3, "use_depth": False, "use_bev": True,
        })


@pytest.fixture
def grid_generator():
    for dependency in ("rosbag", "rospy", "tf2_ros", "cv_bridge", "unidepth_ros.depth_estimator"):
        pytest.importorskip(dependency)
    path = Path(__file__).resolve().parents[1] / "scripts/add_bev_grid_to_dataset.py"
    spec = importlib.util.spec_from_file_location("bev_grid_generator_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bev_occupancy_grid


def test_grid_counts_heights_boundaries_and_invalid_points(grid_generator):
    args = SimpleNamespace(map_width_m=4., minimum_forward_m=0., maximum_forward_m=4.,
                           minimum_height_m=0.05, maximum_height_m=1.6,
                           forward_bins=2, lateral_bins=2, border_margin_ratio=0.)
    points = np.array([[[0., -2., .2], [1., -1., .6], [4., 2., 1.6],
                        [2., 0., .4], [2., 0., .8], [np.nan, 0., .5],
                        [2., 0., 2.], [2., 3., .5]]], dtype=np.float32)
    mask = np.ones((1, 8), dtype=bool)
    mask[0, 4] = False
    grid = grid_generator(points, mask, args)
    np.testing.assert_array_equal(grid[..., 0], [[2, 0], [0, 2]])
    np.testing.assert_allclose(grid[..., 1], [[.4, 0], [0, 1.]])
    assert grid.dtype == np.float32
    np.testing.assert_array_equal(grid_generator(points, np.zeros_like(mask), args), 0)


def test_grid_excludes_image_border(grid_generator):
    args = SimpleNamespace(map_width_m=4., minimum_forward_m=0., maximum_forward_m=4.,
                           minimum_height_m=0.05, maximum_height_m=1.6,
                           forward_bins=2, lateral_bins=2, border_margin_ratio=.25)
    points = np.tile([1., -1., .5], (4, 4, 1))
    grid = grid_generator(points, np.ones((4, 4), dtype=bool), args)
    assert grid[..., 0].sum() == 4
    assert grid[0, 0, 1] == pytest.approx(.5)
