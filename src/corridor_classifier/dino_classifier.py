from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Tuple

import timm
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class Prediction:
    class_index: int
    confidence: float
    probabilities: Tuple[float, ...]
    feature_map: Image.Image = None


def resolve_device(device_name: str) -> torch.device:
    normalized = str(device_name).strip().lower()
    if normalized in ("", "auto"):
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device was requested but CUDA is unavailable: {device_name}"
        )
    return torch.device(normalized)


def build_transform(input_size: List[int]):
    height, width = (int(input_size[0]), int(input_size[1]))
    return transforms.Compose(
        [
            # Stretching preserves the full camera field of view. Training must
            # use the same preprocessing.
            transforms.Resize(
                (height, width),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def visualize_patch_features(
    features: torch.Tensor,
    grid_size: Tuple[int, int],
    num_prefix_tokens: int,
    output_size: Tuple[int, int],
) -> Image.Image:
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError(
            "features must have shape (1, tokens, channels), got "
            f"{tuple(features.shape)}"
        )
    grid_height, grid_width = (int(grid_size[0]), int(grid_size[1]))
    patch_count = grid_height * grid_width
    patch_features = features[
        0,
        int(num_prefix_tokens) : int(num_prefix_tokens) + patch_count,
    ]
    if patch_features.shape[0] != patch_count:
        raise ValueError(
            f"expected {patch_count} patch tokens, got "
            f"{patch_features.shape[0]}"
        )
    if patch_features.shape[1] < 3:
        raise ValueError("patch features must contain at least three channels")

    patch_features = patch_features.detach().float().cpu()
    patch_features = patch_features - patch_features.mean(dim=0, keepdim=True)
    eigenvalues, eigenvectors = torch.linalg.eigh(
        patch_features @ patch_features.transpose(0, 1)
    )
    eigenvalues = eigenvalues[-3:].flip(0).clamp_min(0.0)
    eigenvectors = eigenvectors[:, -3:].flip(1)
    projected = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(0)

    # PCA signs are arbitrary. Fix each sign using its largest-magnitude patch
    # to avoid unnecessary color inversions between frames.
    for channel in range(3):
        values = projected[:, channel]
        anchor = int(torch.argmax(torch.abs(values)))
        if values[anchor] < 0:
            projected[:, channel] = -values

    lower = torch.quantile(projected, 0.02, dim=0, keepdim=True)
    upper = torch.quantile(projected, 0.98, dim=0, keepdim=True)
    projected = (projected - lower) / (upper - lower).clamp_min(1e-6)
    projected = projected.clamp(0.0, 1.0)
    rgb = (
        projected.reshape(grid_height, grid_width, 3) * 255.0
    ).to(torch.uint8)
    feature_map = Image.fromarray(rgb.numpy(), mode="RGB")
    return feature_map.resize(
        (int(output_size[0]), int(output_size[1])),
        Image.Resampling.NEAREST,
    )


def visualize_spatial_features(
    features: torch.Tensor,
    output_size: Tuple[int, int],
) -> Image.Image:
    if features.ndim != 4 or features.shape[0] != 1:
        raise ValueError(
            "features must have shape (1, channels, height, width), got "
            f"{tuple(features.shape)}"
        )
    grid_height = int(features.shape[2])
    grid_width = int(features.shape[3])
    tokens = features.permute(0, 2, 3, 1).reshape(
        1,
        grid_height * grid_width,
        int(features.shape[1]),
    )
    return visualize_patch_features(
        features=tokens,
        grid_size=(grid_height, grid_width),
        num_prefix_tokens=0,
        output_size=output_size,
    )


def create_dino_model(
    model_name: str,
    input_size: List[int],
    num_classes: int,
    pretrained: bool = False,
    pretrained_weights_path: str = None,
) -> nn.Module:
    height, width = (int(input_size[0]), int(input_size[1]))
    img_size = height if height == width else (height, width)
    create_kwargs = {
        "pretrained": bool(pretrained),
        "img_size": img_size,
        "num_classes": int(num_classes),
    }
    if pretrained_weights_path is not None:
        if not pretrained:
            raise ValueError(
                "pretrained_weights_path requires pretrained=True"
            )
        create_kwargs["pretrained_cfg_overlay"] = {
            "file": str(pretrained_weights_path)
        }
    return timm.create_model(model_name, **create_kwargs)


def create_pretrained_feature_model(
    model_name: str,
    pretrained_weights_path: str = None,
) -> nn.Module:
    create_kwargs = {
        "pretrained": True,
        "num_classes": 0,
    }
    if pretrained_weights_path is not None:
        create_kwargs["pretrained_cfg_overlay"] = {
            "file": str(pretrained_weights_path)
        }
    return timm.create_model(model_name, **create_kwargs)


def _is_state_dict(value) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(torch.is_tensor(tensor) for tensor in value.values())
    )


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if _is_state_dict(checkpoint):
        state_dict = dict(checkpoint)
    elif isinstance(checkpoint, Mapping):
        state_dict = None
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if _is_state_dict(candidate):
                state_dict = dict(candidate)
                break
        if state_dict is None:
            raise ValueError(
                "checkpoint must be a state dictionary or contain "
                "'model_state_dict', 'state_dict', or 'model'"
            )
    else:
        raise ValueError("checkpoint must be a mapping")

    for prefix in ("module.", "model.", "dino."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix) :]: value for key, value in state_dict.items()
            }
    return state_dict


def load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch versions before weights_only was added.
        return torch.load(path, map_location="cpu")


class DINOClassifier:
    def __init__(
        self,
        model_config: Dict,
        checkpoint_path: str,
    ):
        self.class_names = tuple(str(name) for name in model_config["class_names"])
        self.num_classes = int(model_config["num_classes"])
        self.device = resolve_device(model_config.get("device", "auto"))
        self.use_fp16 = bool(model_config.get("use_fp16", True)) and (
            self.device.type == "cuda"
        )
        self.transform = build_transform(model_config["input_size"])

        self.model = create_dino_model(
            model_name=str(model_config["model_name"]),
            input_size=model_config["input_size"],
            num_classes=self.num_classes,
            pretrained=False,
        )
        checkpoint = load_checkpoint(checkpoint_path)
        self._validate_checkpoint_metadata(checkpoint)
        state_dict = extract_state_dict(checkpoint)
        self.model.load_state_dict(
            state_dict,
            strict=bool(model_config.get("strict_checkpoint", True)),
        )
        self.model.to(self.device)
        self.model.eval()

    def _validate_checkpoint_metadata(self, checkpoint) -> None:
        if not isinstance(checkpoint, Mapping):
            return
        checkpoint_classes = checkpoint.get("class_names")
        if checkpoint_classes is not None:
            checkpoint_classes = tuple(str(name) for name in checkpoint_classes)
            if checkpoint_classes != self.class_names:
                raise ValueError(
                    "checkpoint class_names do not match config/model.yaml"
                )

    def predict(
        self,
        image: Image.Image,
        include_feature_map: bool = False,
    ) -> Prediction:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        input_tensor = input_tensor.to(self.device, non_blocking=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_fp16
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            feature_map = None
            if include_feature_map:
                features = self.model.forward_features(input_tensor)
                logits = self.model.forward_head(features)
                feature_map = visualize_patch_features(
                    features=features,
                    grid_size=self.model.patch_embed.grid_size,
                    num_prefix_tokens=self.model.num_prefix_tokens,
                    output_size=(
                        int(input_tensor.shape[-1]),
                        int(input_tensor.shape[-2]),
                    ),
                )
            else:
                logits = self.model(input_tensor)
            if logits.ndim != 2 or logits.shape != (1, self.num_classes):
                raise RuntimeError(
                    "model output must have shape "
                    f"(1, {self.num_classes}), got {tuple(logits.shape)}"
                )
            probabilities = torch.softmax(logits.float(), dim=1)[0]

        class_index = int(torch.argmax(probabilities).item())
        probability_values = tuple(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        return Prediction(
            class_index=class_index,
            confidence=probability_values[class_index],
            probabilities=probability_values,
            feature_map=feature_map,
        )


class PretrainedViTFeatureExtractor:
    def __init__(
        self,
        model_name: str,
        input_size: List[int],
        device_name: str = "auto",
        use_fp16: bool = True,
        weights_path: str = None,
    ):
        self.device = resolve_device(device_name)
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        self.transform = build_transform(input_size)
        self.model = create_dino_model(
            model_name=str(model_name),
            input_size=input_size,
            num_classes=0,
            pretrained=True,
            pretrained_weights_path=weights_path,
        )
        if not hasattr(self.model, "forward_features"):
            raise ValueError(f"model does not expose forward_features: {model_name}")
        if not hasattr(self.model, "patch_embed"):
            raise ValueError(f"model does not expose patch_embed: {model_name}")
        if not hasattr(self.model, "num_prefix_tokens"):
            raise ValueError(
                f"model does not expose num_prefix_tokens: {model_name}"
            )
        self.model.to(self.device)
        self.model.eval()

    def extract(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_fp16
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            features = self.model.forward_features(input_tensor)
            return visualize_patch_features(
                features=features,
                grid_size=self.model.patch_embed.grid_size,
                num_prefix_tokens=self.model.num_prefix_tokens,
                output_size=(
                    int(input_tensor.shape[-1]),
                    int(input_tensor.shape[-2]),
                ),
            )


class PretrainedCNNFeatureExtractor:
    def __init__(
        self,
        model_name: str,
        input_size: List[int],
        device_name: str = "auto",
        use_fp16: bool = True,
        weights_path: str = None,
    ):
        self.device = resolve_device(device_name)
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        self.transform = build_transform(input_size)
        self.model = create_pretrained_feature_model(
            model_name=str(model_name),
            pretrained_weights_path=weights_path,
        )
        if not hasattr(self.model, "forward_features"):
            raise ValueError(f"model does not expose forward_features: {model_name}")
        self.model.to(self.device)
        self.model.eval()

    def extract(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_fp16
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            features = self.model.forward_features(input_tensor)
            return visualize_spatial_features(
                features=features,
                output_size=(
                    int(input_tensor.shape[-1]),
                    int(input_tensor.shape[-2]),
                ),
            )
