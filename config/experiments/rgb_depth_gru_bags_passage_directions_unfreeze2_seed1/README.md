# Passage-direction final-two-block unfreeze comparison

This experiment keeps the adopted seed-1 RGB-depth-GRU condition fixed and
changes only the DINOv2 backbone schedule:

- epochs 1-4: backbone frozen
- epochs 5-16: final two DINOv2 blocks and norm unfrozen
- backbone learning rate: `1e-6` with the same cosine schedule

## Result

The table compares each condition at its best `test_passage_macro_f1` epoch.
The `test` split is used for checkpoint selection in this experiment, so these
numbers are controlled selection results rather than an untouched estimate of
generalization.

| Condition | Epoch | Train passage macro-F1 | Test passage macro-F1 | Direction macro-F1 | Front F1 | Left F1 | Right F1 | Exact match | Turning F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Backbone frozen | 8 | 0.6604 | **0.6394** | **0.6363** | 0.9438 | 0.4235 | **0.5417** | 0.7418 | **0.6486** |
| Final two blocks unfrozen | 7 | 0.6951 | 0.6301 | 0.6305 | **0.9464** | 0.4235 | 0.5217 | **0.7484** | 0.6286 |

Unfreezing increased the train/test passage-F1 gap from 0.0210 to 0.0651 and
did not improve the selected passage score. The adopted runtime checkpoint
therefore remains the fully frozen seed-1 model.

Artifacts are written to:

```text
runs/experiments/rgb_depth_gru_bags_passage_directions_unfreeze2_seed1/
weights/experiments/rgb_depth_gru_bags_passage_directions_unfreeze2_seed1/
```
