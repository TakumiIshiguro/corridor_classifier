# Passage-direction augmentation ablation

## Conditions

The adopted frozen RGB-depth-GRU condition was trained with and without all
configured augmentation. Architecture, labels, sequence sampling, optimizer,
thresholds, epochs, and DINOv2 freezing were kept fixed. Seeds 0 and 1 were
both compared.

The `test` split is used for checkpoint selection in these experiments. The
results are therefore controlled selection results, not an untouched estimate
of generalization.

## Results

Each row uses the best `test_passage_macro_f1` epoch for that run.

| Augmentation | Seed | Epoch | Train passage macro-F1 | Test passage macro-F1 | Direction macro-F1 | Exact match | Turning F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Enabled | 0 | 14 | 0.7152 | **0.6212** | **0.6348** | **0.7287** | **0.5806** |
| Disabled | 0 | 13 | 0.7247 | 0.6047 | 0.6187 | 0.7265 | 0.5625 |
| Enabled | 1 | 8 | 0.6604 | 0.6394 | 0.6363 | 0.7418 | **0.6486** |
| Disabled | 1 | 8 | 0.6707 | **0.6440** | **0.6425** | **0.7505** | **0.6486** |

| Two-seed mean | Enabled | Disabled |
| --- | ---: | ---: |
| Passage macro-F1 | **0.6303** | 0.6243 |
| Direction macro-F1 | **0.6356** | 0.6306 |
| Exact match | 0.7352 | **0.7385** |
| Turning F1 | **0.6146** | 0.6056 |

Removing augmentation slightly improved seed 1 but degraded seed 0 more. It
also widened the passage-score range across seeds from 0.0182 to 0.0394. The
adopted augmented seed-1 checkpoint is retained. A more appropriate follow-up
is to sample augmentation parameters once per temporal sequence and apply the
same transform to all three RGB/depth frames, instead of removing augmentation
entirely.

No-augmentation artifacts are written to:

```text
runs/experiments/rgb_depth_gru_bags_passage_directions_no_augmentation_seed0/
runs/experiments/rgb_depth_gru_bags_passage_directions_no_augmentation_seed1/
weights/experiments/rgb_depth_gru_bags_passage_directions_no_augmentation_seed0/
weights/experiments/rgb_depth_gru_bags_passage_directions_no_augmentation_seed1/
```
