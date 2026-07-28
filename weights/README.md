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

The ROS inference node accepts:

- a plain PyTorch state dictionary;
- a dictionary containing `model_state_dict`;
- a dictionary containing `state_dict`.

The model must use the same `model_name`, `input_size`, class order, and
number of classes as `config/model.yaml`. The training script records those
values in the checkpoint so that inference can validate the class order.

Weights are not distributed by this package.
