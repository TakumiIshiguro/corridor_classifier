# RGB-depth-GRU one-second interval comparison

## Conditions

All experiments use the same RGB-depth-GRU architecture, official pretrained
DINOv2 ViT-S/14 backbone, optimizer, augmentation, class weighting, and random
seed. The 4 Hz source data is reused without extracting the bags again.

- `frame_stride: 4`: adjacent GRU inputs are one second apart
- `train_sequence_step: 4`: training windows advance by one second
- `test_sequence_step: 4`: checkpoint-selection windows advance by one second
- 12 epochs; backbone frozen for epochs 1-3, then final two blocks unfrozen

## Results

| GRU frames | Time span | Train sequences | Selection macro-F1 | Full-test accuracy | Full-test macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 s | 1512 | **0.3449** | 0.5333 | **0.3502** |
| 5 | 4 s | 1506 | 0.3050 | 0.5380 | 0.3009 |
| 7 | 6 s | 1500 | 0.2587 | **0.5458** | 0.2578 |

The selected one-second-interval model is the three-frame condition at epoch
9. It needs nine consecutive 4 Hz messages to build the samples at offsets
`[0, 4, 8]`, then continues producing predictions at 4 Hz.

```text
weights/experiments/rgb_depth_gru_stride4/seq3.pth
```

Run it with:

```bash
source /home/takumi/catkin_ws/devel/setup.bash
roslaunch corridor_classifier corridor_classifier.launch \
  config_dir:=/home/takumi/catkin_ws/src/corridor_classifier/config/experiments/rgb_depth_gru_bags_stride4_seq3
```

The previous dense ten-frame model still has the higher macro-F1 (`0.3858`)
but lower accuracy (`0.4585`). Its 2.25-second time span is close to the new
three-frame model's two-second span. These results suggest that extending the
history to four or six seconds introduces stale context, while dense frames
within roughly two seconds retain some useful information.

As before, `cross_road` has no examples, and `dead_end` and `3_way_right`
remain nearly unrecognized. Changing the temporal length alone does not solve
those data and label-distribution problems.
