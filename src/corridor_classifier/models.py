from collections import deque
from contextlib import nullcontext
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import cv2
import torch
from PIL import Image
from torch import nn

from corridor_classifier.dino_classifier import (
    Prediction,
    build_transform,
    create_dino_model,
    extract_state_dict,
    load_checkpoint,
    resolve_device,
)


ARCHITECTURES = ("rgb", "rgb_gru", "rgb_depth", "rgb_depth_gru")


class DepthEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, int(output_dim))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(depth).flatten(1))


class RGBModel(nn.Module):
    """Legacy-compatible single-frame DINO classifier."""

    def __init__(self, dino: nn.Module):
        super().__init__()
        self.dino = dino

    def forward(self, inputs) -> torch.Tensor:
        rgb = inputs["rgb"] if isinstance(inputs, dict) else inputs
        if rgb.ndim == 5:
            rgb = rgb[:, -1]
        return self.dino(rgb)

    def task_parameters(self) -> Iterable[nn.Parameter]:
        return self.dino.head.parameters()


class MultimodalCorridorModel(nn.Module):
    def __init__(
        self,
        dino: nn.Module,
        num_classes: int,
        use_depth: bool,
        use_gru: bool,
        depth_feature_dim: int = 128,
        fusion_dim: int = 256,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 1,
    ):
        super().__init__()
        self.dino = dino
        self.use_depth = bool(use_depth)
        self.use_gru = bool(use_gru)
        rgb_dim = int(getattr(dino, "num_features"))
        self.depth_encoder = (
            DepthEncoder(depth_feature_dim) if self.use_depth else None
        )
        combined_dim = rgb_dim + (int(depth_feature_dim) if self.use_depth else 0)
        self.fusion = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, int(fusion_dim)),
            nn.GELU(),
        )
        if self.use_gru:
            self.gru = nn.GRU(
                input_size=int(fusion_dim),
                hidden_size=int(gru_hidden_size),
                num_layers=int(gru_num_layers),
                batch_first=True,
            )
            classifier_dim = int(gru_hidden_size)
        else:
            self.gru = None
            classifier_dim = int(fusion_dim)
        self.classifier = nn.Linear(classifier_dim, int(num_classes))

    def encode_frames(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rgb.ndim != 4:
            raise ValueError("rgb frames must have shape NxCxHxW")
        rgb_features = self.dino(rgb)
        features = [rgb_features]
        if self.use_depth:
            if depth is None:
                raise ValueError("depth input is required by this architecture")
            if depth.ndim != 4 or depth.shape[0] != rgb.shape[0]:
                raise ValueError("depth frames must match the rgb batch")
            features.append(self.depth_encoder(depth))
        return self.fusion(torch.cat(features, dim=-1))

    def classify_features(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim == 2:
            fused = fused.unsqueeze(1)
        if self.gru is not None:
            fused, _ = self.gru(fused)
        return self.classifier(fused[:, -1])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        rgb = inputs["rgb"]
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)
        if rgb.ndim != 5:
            raise ValueError("rgb must have shape BxTxCxHxW")
        batch_size, sequence_length = rgb.shape[:2]
        depth = inputs.get("depth")
        if depth is not None and depth.ndim == 4:
            depth = depth.unsqueeze(1)
        if self.use_depth and (
            depth is None or depth.shape[:2] != (batch_size, sequence_length)
        ):
            raise ValueError("rgb and depth sequence dimensions must match")
        fused = self.encode_frames(
            rgb.flatten(0, 1),
            depth.flatten(0, 1) if depth is not None else None,
        ).reshape(batch_size, sequence_length, -1)
        return self.classify_features(fused)

    def task_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("dino."):
                yield parameter


def create_corridor_model(
    model_config: Dict,
    pretrained: bool = False,
    pretrained_weights_path: str = None,
) -> nn.Module:
    architecture = str(model_config["architecture"])
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
    common = {
        "model_name": str(model_config["model_name"]),
        "input_size": model_config["input_size"],
        "pretrained": bool(pretrained),
        "pretrained_weights_path": pretrained_weights_path,
    }
    if architecture == "rgb":
        return RGBModel(
            create_dino_model(
                num_classes=int(model_config["num_classes"]),
                **common,
            )
        )
    dino = create_dino_model(num_classes=0, **common)
    return MultimodalCorridorModel(
        dino=dino,
        num_classes=int(model_config["num_classes"]),
        use_depth=bool(model_config["use_depth"]),
        use_gru=bool(model_config["use_gru"]),
        depth_feature_dim=int(model_config.get("depth_feature_dim", 128)),
        fusion_dim=int(model_config.get("fusion_dim", 256)),
        gru_hidden_size=int(model_config.get("gru_hidden_size", 256)),
        gru_num_layers=int(model_config.get("gru_num_layers", 1)),
    )


def depth_to_tensor(
    depth_meters: np.ndarray,
    minimum_m: float,
    maximum_m: float,
) -> torch.Tensor:
    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth map must be two-dimensional")
    valid = np.isfinite(depth) & (depth > 0.0)
    clipped = np.clip(depth, float(minimum_m), float(maximum_m))
    denominator = np.log(float(maximum_m)) - np.log(float(minimum_m))
    normalized = (np.log(clipped) - np.log(float(minimum_m))) / denominator
    normalized[~valid] = 0.0
    return torch.from_numpy(
        np.stack((normalized, valid.astype(np.float32)), axis=0)
    )


def load_model_checkpoint(model, model_config: Dict, path: str) -> None:
    checkpoint = load_checkpoint(path)
    if isinstance(checkpoint, dict):
        checkpoint_architecture = checkpoint.get("architecture")
        if checkpoint_architecture is not None and (
            str(checkpoint_architecture) != str(model_config["architecture"])
        ):
            raise ValueError("checkpoint architecture does not match model.yaml")
        checkpoint_classes = checkpoint.get("class_names")
        if checkpoint_classes is not None and list(checkpoint_classes) != list(
            model_config["class_names"]
        ):
            raise ValueError("checkpoint class_names do not match model.yaml")
    state_dict = extract_state_dict(checkpoint)
    if isinstance(model, RGBModel) and not any(
        key.startswith("dino.") for key in state_dict
    ):
        state_dict = {f"dino.{key}": value for key, value in state_dict.items()}
    model.load_state_dict(
        state_dict,
        strict=bool(model_config.get("strict_checkpoint", True)),
    )


class CorridorPredictor:
    def __init__(self, model_config: Dict, checkpoint_path: str):
        self.model_config = dict(model_config)
        self.class_names = tuple(model_config["class_names"])
        self.device = resolve_device(model_config.get("device", "auto"))
        self.use_fp16 = bool(model_config.get("use_fp16", True)) and (
            self.device.type == "cuda"
        )
        self.transform = build_transform(model_config["input_size"])
        self.sequence_length = int(model_config["sequence_length"])
        self.maximum_gap_seconds = float(
            model_config.get("maximum_gap_seconds", 0.4)
        )
        self.use_depth = bool(model_config["use_depth"])
        self._rgb = deque(maxlen=self.sequence_length)
        self._depth = deque(maxlen=self.sequence_length)
        self._features = deque(maxlen=self.sequence_length)
        self._last_stamp = None
        self.model = create_corridor_model(model_config)
        load_model_checkpoint(self.model, model_config, checkpoint_path)
        self.model.to(self.device).eval()

    def reset(self) -> None:
        self._rgb.clear()
        self._depth.clear()
        self._features.clear()
        self._last_stamp = None

    @property
    def context_length(self) -> int:
        if isinstance(self.model, MultimodalCorridorModel):
            return len(self._features)
        return len(self._rgb)

    def predict(
        self,
        image: Image.Image,
        depth_meters: Optional[np.ndarray] = None,
        stamp: Optional[float] = None,
    ) -> Optional[Prediction]:
        if stamp is not None and self._last_stamp is not None:
            gap = float(stamp) - self._last_stamp
            if gap < 0.0 or gap > self.maximum_gap_seconds:
                self.reset()
        if stamp is not None:
            self._last_stamp = float(stamp)
        rgb_tensor = self.transform(image.convert("RGB"))
        if self.use_depth:
            if depth_meters is None:
                raise ValueError("depth input is required by this architecture")
            depth_meters = np.asarray(depth_meters, dtype=np.float32)
            target_height, target_width = self.model_config["input_size"]
            if depth_meters.shape != (target_height, target_width):
                depth_meters = cv2.resize(
                    depth_meters,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            depth_tensor = depth_to_tensor(
                depth_meters,
                self.model_config["depth_min_m"],
                self.model_config["depth_max_m"],
            )
        else:
            depth_tensor = None
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.use_fp16
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            if isinstance(self.model, MultimodalCorridorModel):
                feature = self.model.encode_frames(
                    rgb_tensor.unsqueeze(0).to(
                        self.device, non_blocking=True
                    ),
                    (
                        depth_tensor.unsqueeze(0).to(
                            self.device, non_blocking=True
                        )
                        if depth_tensor is not None
                        else None
                    ),
                )[0]
                self._features.append(feature)
                if len(self._features) < self.sequence_length:
                    return None
                logits = self.model.classify_features(
                    torch.stack(tuple(self._features)).unsqueeze(0)
                )
            else:
                self._rgb.append(rgb_tensor)
                logits = self.model(
                    {
                        "rgb": torch.stack(tuple(self._rgb))
                        .unsqueeze(0)
                        .to(self.device, non_blocking=True)
                    }
                )
            probabilities = torch.softmax(logits.float(), dim=1)[0]
        values = tuple(float(value) for value in probabilities.cpu().tolist())
        index = int(torch.argmax(probabilities))
        return Prediction(index, values[index], values)
