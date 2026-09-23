"""Checks for rgb_bev_gru training/runtime integration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch

from corridor_classifier.config import load_config
from corridor_classifier.dataset import CorridorMultiInputDataset, CorridorSample
from corridor_classifier.models import CorridorPredictor
from test_bev import TinyDino


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"review_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value,expected", [("", .75), ("1.5", 1.5), (2., 2.)])
def test_bev_ros_parameter_overrides_yaml_only_when_supplied(monkeypatch, value, expected):
    pytest.importorskip("rospy")
    node = load_script("corridor_classifier_node")
    config = load_config()
    config["runtime"]["bev_max_time_difference_seconds"] = .75
    monkeypatch.setattr(node, "rospy", SimpleNamespace(
        get_param=lambda key, default: value if key == "~bev_max_time_difference_seconds" else default))
    node._apply_ros_overrides(config)
    assert config["runtime"]["bev_max_time_difference_seconds"] == expected


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_bev_ros_parameter_rejects_invalid_limits(monkeypatch, value):
    pytest.importorskip("rospy")
    node = load_script("corridor_classifier_node")
    monkeypatch.setattr(node, "rospy", SimpleNamespace(
        get_param=lambda key, default: value if key == "~bev_max_time_difference_seconds" else default))
    with pytest.raises(ValueError):
        node._apply_ros_overrides(load_config())


def test_pointcloud_callback_preserves_counts_stamp_and_empty_cloud(monkeypatch):
    rospy = pytest.importorskip("rospy")
    point_cloud2 = pytest.importorskip("sensor_msgs.point_cloud2")
    from std_msgs.msg import Header
    from corridor_classifier.bev_points import LatestBevGridSubscriber

    monkeypatch.setattr(rospy, "Subscriber", Mock())
    subscriber = LatestBevGridSubscriber("/test", [0.1, 5.1], 7., 50, 70)
    assert subscriber.latest() == (None, None)
    header = Header(stamp=rospy.Time.from_sec(12.), frame_id="base_footprint")
    points = [[1.15, -1.15, 0.]] * 80 + [[1.15, 1.15, 0.]] * 79
    subscriber._callback(point_cloud2.create_cloud_xyz32(header, points))
    grid, stamp = subscriber.latest()
    assert stamp == 12.
    assert grid.shape == (50, 70, 2)
    assert grid[..., 0].sum() == 159
    assert sorted(grid[..., 0][grid[..., 0] > 0]) == [79, 80]
    assert not grid[..., 1].any()
    grid[:] = 0
    assert subscriber.latest()[0][..., 0].sum() == 159
    subscriber._callback(point_cloud2.create_cloud_xyz32(header, []))
    assert not subscriber.latest()[0].any()


@pytest.mark.parametrize("output_mode", ["class", "passage_directions"])
@pytest.mark.parametrize("drop_height", [False, True])
def test_bev_predictor_matches_dataset_and_batched_gru(
    tmp_path, monkeypatch, output_mode, drop_height
):
    monkeypatch.setattr("corridor_classifier.models.create_dino_model", TinyDino)
    monkeypatch.setattr("corridor_classifier.models.load_model_checkpoint", lambda *args: None)
    config = dict(load_config()["model"], output_mode=output_mode, device="cpu",
                  use_fp16=False, input_size=[16, 16], dino_readout="last_cls",
                  bev_drop_height=drop_height, bev_feature_dim=8, fusion_dim=10,
                  gru_hidden_size=7)
    predictor = CorridorPredictor(config, "unused.pth")
    samples, images, grids = [], [], []
    for index in range(9):
        image = Image.new("RGB", (16, 16), (index * 20, 60, 30))
        grid = np.zeros((50, 70, 2), np.float32)
        grid[..., 0] = np.arange(70) * 2 + index
        grid[..., 1] = .7
        image_path, grid_path = tmp_path / f"{index}.png", tmp_path / f"{index}.npy"
        image.save(image_path)
        np.save(grid_path, grid)
        samples.append(CorridorSample(str(image_path), 0, "session", index * .25,
                                      bev_grid_path=str(grid_path)))
        images.append(image)
        grids.append(grid)
    dataset = CorridorMultiInputDataset(samples, [16, 16], config)
    inputs, _ = dataset[0]
    for position, index in enumerate((0, 4, 8)):
        torch.testing.assert_close(predictor._bev_to_tensor(grids[index]), inputs["bev"][position])
    results = [predictor.predict(image, stamp=index * .25, bev_grid=grids[index])
               for index, image in enumerate(images)]
    assert all(result is None for result in results[:8])
    assert results[-1] is not None
    with torch.inference_mode():
        output = predictor.model({key: tensor.unsqueeze(0) for key, tensor in inputs.items()})
    if output_mode == "class":
        expected = output.softmax(-1)[0].numpy()
        actual = results[-1].probabilities
    else:
        expected = output["direction_logits"].sigmoid()[0].numpy()
        actual = results[-1].direction_probabilities
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert predictor.predict(images[0], stamp=10., bev_grid=grids[0]) is None
    assert predictor.context_length == 1


def test_holdout_evaluation_respects_bev_manifest_column(monkeypatch):
    script = load_script("evaluate_session_holdout")
    config = {"model": dict(load_config()["model"]),
              "dataset": {"test_data_dir": "/tmp/unused"}, "training": {}}
    monkeypatch.setattr(script, "parse_args", lambda: SimpleNamespace(
        config_dir="unused", data_dir="", session_names=["session"]))
    monkeypatch.setattr(script, "load_training_config", lambda _: config)
    observed = {}

    class StopAfterDatasetLoad(Exception):
        pass

    def capture(*args, **kwargs):
        observed.update(kwargs)
        raise StopAfterDatasetLoad

    monkeypatch.setattr(script, "load_dataset_samples", capture)
    with pytest.raises(StopAfterDatasetLoad):
        script.main()
    assert observed.get("bev_grid_column") == config["model"]["bev_manifest_column"]


@pytest.mark.parametrize("max_time_difference", [0.5, 1.0])
@pytest.mark.parametrize("bev_offset,expected_calls", [
    (0., 1), (-1., 1), (1., 1), (-1.001, 0), (1.001, 0),
    (-9., 0), (None, 0), (float("nan"), 0), (float("inf"), 0),
    ("missing_grid", 0), ((0., -9., 0.), 2),
])
def test_ros_node_does_not_infer_with_stale_bev(
    monkeypatch, max_time_difference, bev_offset, expected_calls
):
    for dependency in ("rospy", "cv_bridge", "scenario_navigation_msgs.msg"):
        pytest.importorskip(dependency)
    node = load_script("corridor_classifier_node")
    debouncer = Mock()
    monkeypatch.setattr(node, "ConsecutiveConfirmDebouncer", lambda **kwargs: debouncer)
    config = load_config()
    # Test the configured limit independently of the deployed YAML value.
    # Numeric offsets are multiples of this limit relative to the RGB stamp.
    config["runtime"]["bev_max_time_difference_seconds"] = max_time_difference
    monkeypatch.setattr(node, "load_config", lambda _: config)
    monkeypatch.setattr(node, "_apply_ros_overrides", lambda _: None)
    # The file itself is covered separately by the real checkpoint smoke test.
    monkeypatch.setattr(node.os.path, "isfile", lambda _: True)
    offsets = bev_offset if isinstance(bev_offset, tuple) else (bev_offset,)
    stamps = tuple(
        10. + offset * max_time_difference
        if isinstance(offset, (int, float)) else offset
        for offset in offsets
    )
    stopping = iter([False] * len(stamps) + [True])
    ros = SimpleNamespace(init_node=Mock(), get_param=lambda key, default: default,
                          Publisher=Mock(), Rate=Mock(), loginfo=Mock(),
                          loginfo_throttle=Mock(), logwarn_throttle=Mock(),
                          is_shutdown=lambda: next(stopping),
                          Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 10.)),
                          get_time=lambda: 10.)
    monkeypatch.setattr(node, "rospy", ros)
    classifier = SimpleNamespace(
        class_names=config["model"]["class_names"], output_mode="passage_directions",
        use_depth=False, use_bev=True, device="cpu", sequence_length=3, frame_stride=4,
        context_length=0, required_context_length=9, predict=Mock(return_value=None),
        reset=Mock(),
    )
    monkeypatch.setattr(node, "CorridorPredictor", lambda *args: classifier)
    monkeypatch.setattr(node, "CmdDirTurningGate", lambda **kwargs: SimpleNamespace(is_turning=lambda: False))
    monkeypatch.setattr(node, "ScenarioTargetLabelsSubscriber", lambda **kwargs: Mock())
    message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 10.)))
    monkeypatch.setattr(node, "LatestImageSubscriber", lambda _: SimpleNamespace(take_latest=lambda: message))
    readings = iter([
        (None, None) if stamp == "missing_grid" else
        (np.zeros((50, 70, 2), np.float32), stamp)
        for stamp in stamps
    ])
    monkeypatch.setattr(node, "LatestBevGridSubscriber", lambda *args: SimpleNamespace(
        latest=lambda: next(readings)))
    calls = Mock()
    calls.attach_mock(classifier.predict, "predict")
    calls.attach_mock(classifier.reset, "reset")
    monkeypatch.setattr(node, "CvBridge", lambda: SimpleNamespace(
        imgmsg_to_cv2=lambda *args, **kwargs: np.zeros((16, 16, 3), np.uint8)))
    node.main()
    assert classifier.predict.call_count == expected_calls
    assert classifier.reset.call_count == len(stamps) - expected_calls
    assert debouncer.reset.call_count == len(stamps) - expected_calls
    if len(stamps) > 1:
        assert [call[0] for call in calls.mock_calls] == ["predict", "reset", "predict"]
