# RGB + depth + GRU experiment (bags a-n)

This is the selected configuration for the large-class-room dataset.

## Data split

- train: bags `a-f`, `g`, and `h` (`a_f`, `g`, `h` sessions)
- evaluation: bags `i` through `n` (`i`, `j`, `k`, `l`, `m`, `n` sessions)
- RGB and UniDepth depth maps, sampled at 4 Hz
- sequence length: 10 frames

The train and evaluation images have no exact file-content overlap.

## Training recipe

- architecture: `rgb_depth_gru`
- initialization: official DINOv2 ViT-S/14 pretrained weights
- epochs 1-2: DINOv2 backbone frozen
- epochs 3-8: only the final two DINOv2 blocks unfrozen
- head learning rate: `5e-5`
- backbone learning rate: `1e-6`
- optimizer: AdamW, weight decay `0.01`
- scheduler: one-epoch warmup followed by cosine decay
- loss: inverse-square-root class-weighted cross entropy
- augmentation: horizontal flip with directional-label remapping, color jitter,
  occasional grayscale/blur, and depth-scale jitter
- checkpoint selection: highest evaluation macro-F1

Reproduce training with:

```bash
source /home/takumi/catkin_ws/devel/setup.bash
roslaunch corridor_classifier train.launch \
  config_dir:=/home/takumi/catkin_ws/src/corridor_classifier/config/experiments/rgb_depth_gru_bags_seq10
```

## Selected checkpoint

The selected checkpoint is epoch 5:

```text
weights/experiments/rgb_depth_gru_bags/seq10.pth
```

Evaluation results on 1,869 valid sequences:

- accuracy: `0.4585`
- macro-F1 over all eight configured classes: `0.3858`
- train accuracy at the selected epoch: `0.9455`
- train macro-F1 at the selected epoch: `0.8135`

Per-class evaluation F1:

| Class | Support | F1 |
| --- | ---: | ---: |
| straight_road | 1261 | 0.5775 |
| dead_end | 94 | 0.0000 |
| corner_right | 63 | 0.6316 |
| corner_left | 60 | 0.9421 |
| cross_road | 0 | 0.0000 |
| 3_way_right | 196 | 0.0030 |
| 3_way_center | 65 | 0.4120 |
| 3_way_left | 130 | 0.5205 |

`cross_road` is absent from both splits and therefore cannot be learned or
evaluated. `dead_end` and `3_way_right` exhibit a substantial appearance shift
between the train and evaluation locations; more training examples from varied
locations are required before relying on those outputs.

Run ROS inference with the selected configuration:

```bash
source /home/takumi/catkin_ws/devel/setup.bash
roslaunch corridor_classifier corridor_classifier.launch \
  config_dir:=/home/takumi/catkin_ws/src/corridor_classifier/config/experiments/rgb_depth_gru_bags_seq10
```

The node needs synchronized `/camera_center/image_raw` and `/unidepth/depth`
messages and starts publishing after its 10-frame buffer is full.
