#!/usr/bin/env python3
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from PIL import Image

from corridor_classifier.config import (
    load_config,
    load_yaml,
    package_root,
    resolve_path,
)
from corridor_classifier.dataset import class_counts, load_dataset_samples
from corridor_classifier.dino_classifier import (
    DINOClassifier,
    PretrainedCNNFeatureExtractor,
    PretrainedViTFeatureExtractor,
)
from corridor_classifier.feature_visualization import (
    make_feature_panel,
    representative_indices,
)


def load_visualization_config(config_dir):
    data = load_yaml(os.path.join(config_dir, "feature_visualization.yaml"))
    config = dict(data.get("feature_visualization", {}))
    if not config.get("data_dir"):
        raise ValueError(
            "feature_visualization.yaml must define "
            "feature_visualization.data_dir"
        )
    if not config.get("output_dir"):
        raise ValueError(
            "feature_visualization.yaml must define "
            "feature_visualization.output_dir"
        )
    config["max_images"] = int(config.get("max_images", 0))
    config["min_images_per_class"] = int(
        config.get("min_images_per_class", 0)
    )
    seeds = config.get("seeds", [])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(
            "feature_visualization.seeds must be a non-empty list"
        )
    config["seeds"] = [int(seed) for seed in seeds]
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("feature_visualization.seeds must be unique")
    for model_key in ("imagenet_vit", "imagenet_resnet"):
        model_config = config.get(model_key, {})
        if not isinstance(model_config, dict):
            raise ValueError(
                f"feature_visualization.{model_key} must be a mapping"
            )
        model_config = dict(model_config)
        if not model_config.get("model_name"):
            raise ValueError(
                f"feature_visualization.{model_key}.model_name is required"
            )
        model_config["model_name"] = str(
            model_config["model_name"]
        ).strip()
        model_config["weights_path"] = str(
            model_config.get("weights_path", "") or ""
        ).strip()
        config[model_key] = model_config
    if config["max_images"] <= 0:
        raise ValueError("feature_visualization.max_images must be positive")
    if config["min_images_per_class"] <= 0:
        raise ValueError(
            "feature_visualization.min_images_per_class must be positive"
        )
    return config


def main():
    rospy.init_node("corridor_feature_visualization")
    config_dir = os.path.abspath(
        os.path.expanduser(
            rospy.get_param(
                "~config_dir",
                os.path.join(package_root(), "config"),
            )
        )
    )
    config = load_config(config_dir)
    visualization = load_visualization_config(config_dir)
    model_config = config["model"]
    checkpoint_path = resolve_path(
        model_config["checkpoint_path"],
        package_root(),
    )
    data_dir = resolve_path(visualization["data_dir"], package_root())
    output_dir = os.path.join(
        resolve_path(visualization["output_dir"], package_root()),
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    os.makedirs(output_dir, exist_ok=False)

    classifier = DINOClassifier(model_config, checkpoint_path)
    imagenet_config = visualization["imagenet_vit"]
    imagenet_weights_path = None
    if imagenet_config["weights_path"]:
        imagenet_weights_path = resolve_path(
            imagenet_config["weights_path"],
            package_root(),
        )
    imagenet_extractor = PretrainedViTFeatureExtractor(
        model_name=imagenet_config["model_name"],
        input_size=model_config["input_size"],
        device_name=model_config["device"],
        use_fp16=model_config["use_fp16"],
        weights_path=imagenet_weights_path,
    )
    rospy.loginfo(
        "loaded ImageNet pretrained feature extractor: %s",
        imagenet_config["model_name"],
    )
    resnet_config = visualization["imagenet_resnet"]
    resnet_weights_path = None
    if resnet_config["weights_path"]:
        resnet_weights_path = resolve_path(
            resnet_config["weights_path"],
            package_root(),
        )
    resnet_extractor = PretrainedCNNFeatureExtractor(
        model_name=resnet_config["model_name"],
        input_size=model_config["input_size"],
        device_name=model_config["device"],
        use_fp16=model_config["use_fp16"],
        weights_path=resnet_weights_path,
    )
    rospy.loginfo(
        "loaded ImageNet pretrained feature extractor: %s",
        resnet_config["model_name"],
    )
    samples = load_dataset_samples(data_dir, model_config["num_classes"])
    counts = class_counts(samples, model_config["num_classes"])
    missing_classes = [
        model_config["class_names"][index]
        for index, count in enumerate(counts)
        if count < visualization["min_images_per_class"]
    ]
    if missing_classes:
        rospy.logwarn(
            "cannot show %d images for missing/insufficient class(es): %s",
            visualization["min_images_per_class"],
            ", ".join(missing_classes),
        )
    class_indices = [sample.class_index for sample in samples]
    saved_count = 0
    for seed in visualization["seeds"]:
        indices = representative_indices(
            class_indices,
            visualization["max_images"],
            visualization["min_images_per_class"],
            seed=seed,
        )
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=False)
        rospy.loginfo(
            "seed=%d: extracting DINOv2, ViT, and ResNet features "
            "from %d/%d images",
            seed,
            len(indices),
            len(samples),
        )

        for output_index, sample_index in enumerate(indices):
            sample = samples[sample_index]
            with Image.open(sample.image_path) as stream:
                source_image = stream.convert("RGB")
            prediction = classifier.predict(
                source_image,
                include_feature_map=True,
            )
            imagenet_feature_map = imagenet_extractor.extract(source_image)
            resnet_feature_map = resnet_extractor.extract(source_image)
            panel = make_feature_panel(
                source_image=source_image,
                feature_map=prediction.feature_map,
                class_name=classifier.class_names[prediction.class_index],
                confidence=prediction.confidence,
                probabilities=prediction.probabilities,
                label_name=classifier.class_names[sample.class_index],
                comparison_feature_map=imagenet_feature_map,
                comparison_name="ImageNet ViT-S/16",
                resnet_feature_map=resnet_feature_map,
                resnet_name="ImageNet ResNet-18",
            )
            filename = (
                f"{output_index:02d}_"
                f"{os.path.splitext(os.path.basename(sample.image_path))[0]}_"
                f"label-{classifier.class_names[sample.class_index]}_"
                f"pred-{classifier.class_names[prediction.class_index]}.png"
            )
            output_path = os.path.join(seed_dir, filename)
            panel.save(output_path, format="PNG")
            saved_count += 1
            rospy.loginfo(
                "seed=%d saved %d/%d: %s label=%s "
                "prediction=%s confidence=%.3f",
                seed,
                output_index + 1,
                len(indices),
                output_path,
                classifier.class_names[sample.class_index],
                classifier.class_names[prediction.class_index],
                prediction.confidence,
            )

    rospy.loginfo(
        "saved %d feature visualization PNG files to %s",
        saved_count,
        output_dir,
    )


if __name__ == "__main__":
    main()
