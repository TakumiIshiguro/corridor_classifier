# Depth modality comparison

## Question

The adopted RGB+Depth+GRU model concatenates 384 RGB features and 128 depth
features before a shared fusion layer. This comparison checks whether the model
uses the matching depth geometry and whether forcing more depth use improves the
held-out bag sessions.

## Baseline diagnosis

Checkpoint:
`weights/experiments/rgb_depth_gru_bags_passage_directions_stable_seed1/model.pth`

- Mean fusion weight norm per input feature: RGB 0.4149, depth 0.4245
  (depth/RGB 1.023).
- Mean fusion output contribution norm: RGB 13.9416, depth 1.7065
  (depth/RGB 0.1224).
- Replacing every depth sequence with a sequence from another test sample changed
  passage macro-F1 from 0.6394 to 0.6300.
- The shuffled-depth direction macro-F1 changed from 0.6363 to 0.6400, while
  turning F1 changed from 0.6486 to 0.6000.

The learned weight per feature is not smaller for depth, but RGB has three times
as many features and dominates the actual fusion activation. More importantly,
matching depth geometry contributes little to the three opening directions.

Zeroing both depth and its validity mask reduced passage macro-F1 to 0.4181, but
this is an out-of-distribution all-invalid depth map and therefore is not evidence
that the model uses spatial depth correctly. Shuffling valid depth sequences is
the more useful diagnostic here.

## RGB modality dropout experiment

Config:
`config/experiments/rgb_depth_gru_bags_passage_directions_rgb_dropout_seed1`

Training used the baseline settings and seed 1, with one change: the complete RGB
sequence was replaced by the neutral normalized input with probability 0.20.
Inference is unchanged.

| Metric | Baseline | RGB dropout 0.20 |
|---|---:|---:|
| Best epoch | 8 | 15 |
| Passage macro-F1 | 0.6394 | 0.6140 |
| Direction macro-F1 | 0.6363 | 0.6327 |
| Direction exact accuracy | 0.7418 | 0.7374 |
| Turning F1 | 0.6486 | 0.5581 |
| Depth/RGB fusion contribution | 0.1224 | 0.1912 |
| Passage macro-F1 with shuffled depth | 0.6300 | 0.5975 |

RGB modality dropout increased the measured depth contribution by about 56%, but
reduced the normal passage macro-F1 by 0.0254. It is therefore not adopted.

## Decision and spatial Depth follow-up

Keep the current runtime checkpoint. A fixed multiplier on depth features is also
not recommended because the following linear layer can learn the inverse scale.

The spatial follow-up was completed by changing the final Depth pooling grid
from 1x1 to 4x4 before projection. Passage macro-F1 improved from 0.6212 to
0.6469 for seed 0 and from 0.6394 to 0.6456 for seed 1. The two-seed mean
improved from 0.6303 to 0.6462, so the 4x4 Depth representation is the next
candidate. Full readout and per-task results are in
`config/experiments/DINOV2_CLASSIFIER_COMPARISON.md`.

These values are controlled comparisons rather than an untouched final estimate:
the current training loop evaluates the test bags every epoch and selects the best
checkpoint using the test passage macro-F1.
