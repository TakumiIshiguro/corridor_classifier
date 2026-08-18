from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
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
from corridor_classifier.passage_directions import (
    class_name_from_directions,
    threshold_directions,
)


ARCHITECTURES = ("rgb", "rgb_gru", "rgb_depth", "rgb_depth_gru")
# (readout_layers, append_patch_mean, regional_grid=(rows, columns))
DINO_READOUTS = {
    "last_cls": (1, False, (0, 0)),
    "last_cls_patch_mean": (1, True, (0, 0)),
    "last4_cls": (4, False, (0, 0)),
    "last4_cls_patch_mean": (4, True, (0, 0)),
    # Averages the last layer's patch tokens within left/center/right image
    # columns instead of collapsing them into one global vector. DINOv2's
    # dense-prediction evaluations (segmentation/depth) read out spatially
    # arranged patch tokens rather than the CLS token because patch tokens
    # carry spatially localized semantics; this mirrors that for a task
    # (front/left/right passage direction) that is inherently about which
    # image region is open.
    "last_cls_regional3": (1, False, (1, 3)),
    "last_cls_regional5": (1, False, (1, 5)),
    "last_cls_regional3x2": (1, False, (2, 3)),
}


def regional_patch_features(
    dino: nn.Module, patch_tokens: torch.Tensor, rows: int, columns: int
) -> torch.Tensor:
    """Average-pools the last layer's patch tokens within a `rows` x
    `columns` grid of image bands, preserving which region of the image each
    feature came from instead of collapsing all patches into one vector.
    """
    if not hasattr(dino, "patch_embed"):
        raise ValueError("DINO model must expose patch_embed for regional readouts")
    grid_height, grid_width = dino.patch_embed.grid_size
    batch_size, num_patches, channels = patch_tokens.shape
    if num_patches != grid_height * grid_width:
        raise ValueError(
            "patch token count does not match patch_embed.grid_size: "
            f"{num_patches} != {grid_height}x{grid_width}"
        )
    grid = patch_tokens.reshape(batch_size, grid_height, grid_width, channels)
    region_means = []
    for row_band in grid.tensor_split(int(rows), dim=1):
        for cell in row_band.tensor_split(int(columns), dim=2):
            region_means.append(cell.reshape(batch_size, -1, channels).mean(dim=1))
    return torch.cat(region_means, dim=-1)


def encode_dino_readout(
    dino: nn.Module, rgb: torch.Tensor, dino_readout: str
) -> torch.Tensor:
    if dino_readout == "last_cls":
        return dino(rgb)
    if dino_readout not in DINO_READOUTS:
        raise ValueError(f"unsupported DINO readout: {dino_readout}")
    if not hasattr(dino, "get_intermediate_layers"):
        raise ValueError(
            f"DINO model must expose get_intermediate_layers for readout {dino_readout}"
        )
    readout_layers, append_patch_mean, regional_grid = DINO_READOUTS[dino_readout]
    outputs = dino.get_intermediate_layers(
        rgb,
        n=readout_layers,
        return_prefix_tokens=True,
        norm=True,
    )
    class_features = torch.cat(
        [prefix_tokens[:, 0] for _, prefix_tokens in outputs],
        dim=-1,
    )
    features = [class_features]
    if append_patch_mean:
        features.append(outputs[-1][0].mean(dim=1))
    regional_rows, regional_columns = regional_grid
    if regional_rows > 0 and regional_columns > 0:
        features.append(
            regional_patch_features(
                dino, outputs[-1][0], regional_rows, regional_columns
            )
        )
    if len(features) == 1:
        return features[0]
    return torch.cat(features, dim=-1)


@dataclass(frozen=True)
class PassagePrediction:
    direction_probabilities: tuple
    open_directions: tuple
    class_name: str


class DepthEncoder(nn.Module):
    def __init__(self, output_dim: int, pool_size: int = 1):
        super().__init__()
        self.pool_size = int(pool_size)
        if self.pool_size <= 0:
            raise ValueError("depth pool size must be positive")
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
            nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size)),
        )
        self.projection = nn.Linear(
            128 * self.pool_size * self.pool_size,
            int(output_dim),
        )

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
        depth_pool_size: int = 1,
        fusion_dim: int = 256,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 1,
        output_mode: str = "class",
        dino_readout: str = "last_cls",
    ):
        super().__init__()
        self.dino = dino
        self.use_depth = bool(use_depth)
        self.use_gru = bool(use_gru)
        self.output_mode = str(output_mode)
        self.dino_readout = str(dino_readout)
        if self.output_mode not in ("class", "passage_directions"):
            raise ValueError(f"unsupported output_mode: {self.output_mode}")
        if self.dino_readout not in DINO_READOUTS:
            raise ValueError(f"unsupported DINO readout: {self.dino_readout}")
        readout_layers, append_patch_mean, regional_grid = DINO_READOUTS[
            self.dino_readout
        ]
        regional_cells = int(regional_grid[0]) * int(regional_grid[1])
        rgb_dim = int(getattr(dino, "num_features")) * (
            readout_layers + int(append_patch_mean) + regional_cells
        )
        self.depth_encoder = (
            DepthEncoder(depth_feature_dim, depth_pool_size)
            if self.use_depth
            else None
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
        if self.output_mode == "passage_directions":
            self.direction_classifier = nn.Linear(classifier_dim, 3)
            self.classifier = None
        else:
            self.direction_classifier = None
            self.classifier = nn.Linear(classifier_dim, int(num_classes))

    def encode_frames(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rgb.ndim != 4:
            raise ValueError("rgb frames must have shape NxCxHxW")
        rgb_features = self._encode_rgb(rgb)
        features = [rgb_features]
        if self.use_depth:
            if depth is None:
                raise ValueError("depth input is required by this architecture")
            if depth.ndim != 4 or depth.shape[0] != rgb.shape[0]:
                raise ValueError("depth frames must match the rgb batch")
            features.append(self.depth_encoder(depth))
        return self.fusion(torch.cat(features, dim=-1))

    def _encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        return encode_dino_readout(self.dino, rgb, self.dino_readout)

    def classify_features(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim == 2:
            fused = fused.unsqueeze(1)
        if self.gru is not None:
            fused, _ = self.gru(fused)
        features = fused[:, -1]
        if self.output_mode == "passage_directions":
            return {"direction_logits": self.direction_classifier(features)}
        return self.classifier(features)

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
    output_mode = str(model_config.get("output_mode", "class"))
    if architecture == "rgb" and output_mode == "class":
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
        depth_pool_size=int(model_config.get("depth_pool_size", 1)),
        fusion_dim=int(model_config.get("fusion_dim", 256)),
        gru_hidden_size=int(model_config.get("gru_hidden_size", 256)),
        gru_num_layers=int(model_config.get("gru_num_layers", 1)),
        output_mode=output_mode,
        dino_readout=str(model_config.get("dino_readout", "last_cls")),
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
        checkpoint_output_mode = checkpoint.get("output_mode")
        if checkpoint_output_mode is not None and str(
            checkpoint_output_mode
        ) != str(model_config.get("output_mode", "class")):
            raise ValueError("checkpoint output_mode does not match model.yaml")
        checkpoint_variant = checkpoint.get("variant", {})
        checkpoint_readout = checkpoint_variant.get("dino_readout")
        if checkpoint_readout is not None and str(checkpoint_readout) != str(
            model_config.get("dino_readout", "last_cls")
        ):
            raise ValueError("checkpoint DINO readout does not match model.yaml")
        checkpoint_depth_pool_size = checkpoint_variant.get("depth_pool_size")
        if checkpoint_depth_pool_size is not None and int(
            checkpoint_depth_pool_size
        ) != int(model_config.get("depth_pool_size", 1)):
            raise ValueError(
                "checkpoint depth pool size does not match model.yaml"
            )
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
        self.output_mode = str(model_config.get("output_mode", "class"))
        self.direction_thresholds = tuple(
            float(value)
            for value in model_config.get("direction_thresholds", [0.5] * 3)
        )
        self.device = resolve_device(model_config.get("device", "auto"))
        self.use_fp16 = bool(model_config.get("use_fp16", True)) and (
            self.device.type == "cuda"
        )
        self.transform = build_transform(model_config["input_size"])
        self.sequence_length = int(model_config["sequence_length"])
        self.frame_stride = int(model_config.get("frame_stride", 1))
        self.required_context_length = (
            (self.sequence_length - 1) * self.frame_stride + 1
        )
        self.maximum_gap_seconds = float(
            model_config.get("maximum_gap_seconds", 0.4)
        )
        self.use_depth = bool(model_config["use_depth"])
        self._rgb = deque(maxlen=self.required_context_length)
        self._depth = deque(maxlen=self.required_context_length)
        self._features = deque(maxlen=self.required_context_length)
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
                if len(self._features) < self.required_context_length:
                    return None
                features = tuple(self._features)[:: self.frame_stride]
                logits = self.model.classify_features(
                    torch.stack(features).unsqueeze(0)
                )
            else:
                self._rgb.append(rgb_tensor)
                if len(self._rgb) < self.required_context_length:
                    return None
                rgb = tuple(self._rgb)[:: self.frame_stride]
                logits = self.model(
                    {
                        "rgb": torch.stack(rgb)
                        .unsqueeze(0)
                        .to(self.device, non_blocking=True)
                    }
                )
            if self.output_mode == "passage_directions":
                direction_probabilities = torch.sigmoid(
                    logits["direction_logits"].float()
                )[0]
            else:
                probabilities = torch.softmax(logits.float(), dim=1)[0]
        if self.output_mode == "passage_directions":
            direction_values = tuple(
                float(value)
                for value in direction_probabilities.cpu().tolist()
            )
            open_directions = threshold_directions(
                direction_values, self.direction_thresholds
            )
            return PassagePrediction(
                direction_probabilities=direction_values,
                open_directions=open_directions,
                class_name=class_name_from_directions(open_directions),
            )
        values = tuple(float(value) for value in probabilities.cpu().tolist())
        index = int(torch.argmax(probabilities))
        return Prediction(index, values[index], values)
