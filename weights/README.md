# Weights

## DINOv2 pretrained backbone

The default training configuration reads the non-register DINOv2 ViT-S/14
backbone from:

```text
weights/dinov2_vits14_pretrain.pth
```

`dinov2_vits14_reg4_pretrain.pth` is not used by the default configuration.
The configured `timm` model name does not contain `reg4`, so it expects the
non-register weights.

## Trained corridor classifier

Training writes the best eight-class checkpoint to:

```text
weights/corridor_classifier.pth
```

The temporal and/or depth variants use:

```text
weights/corridor_classifier_rgb_gru.pth
weights/corridor_classifier_rgb_depth.pth
weights/corridor_classifier_rgb_depth_gru.pth
```

The ROS inference node accepts:

- a plain PyTorch state dictionary;
- a dictionary containing `model_state_dict`;
- a dictionary containing `state_dict`.

The model must use the same `model_name`, `input_size`, class order, and
number of classes and architecture as `config/model.yaml`. The training script
records the temporal, GRU, fusion, and depth normalization settings in the
checkpoint so inference can reject an incompatible model.

Weights are not distributed by this package.
