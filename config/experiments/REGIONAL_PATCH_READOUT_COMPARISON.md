# Regional patch-token readout comparison

## Motivation

DINOv2's own dense-prediction evaluations (semantic segmentation, depth
estimation; see the
[DINOv2 paper](https://ar5iv.labs.arxiv.org/html/2304.07193), Section 7.4
and Tables 10-11) read out spatially arranged patch tokens rather than the
global CLS token, because patch tokens carry spatially localized semantics
(Section 7.5: matched patches correspond to the same object part across
images). The adopted `rgb_depth_gru` model classifies `front`/`left`/`right`
passage directions, a task that is inherently about which side of the image
is open. The existing RGB readout only used the global `last_cls` token,
while the depth encoder already benefits from preserving left/center/right,
near/far layout via `depth_pool_size: 4` (see
`DEPTH_MODALITY_COMPARISON.md`).

## Change

Added a `last_cls_regional3` DINO readout
(`src/corridor_classifier/models.py`). The last transformer layer's patch
tokens are reshaped to the DINOv2 patch grid (16x16 for 224x224 input),
split into left/center/right column bands, and each band is mean-pooled
into its own feature vector. These three regional vectors are concatenated
with the global CLS token before fusion with the depth features. Unlike the
previously rejected `last_cls_patch_mean`/`last4_cls*` readouts (which
collapsed all patch tokens into a single global vector or stacked more CLS
tokens, adding capacity without spatial structure and overfitting the
training bags — see `DINOV2_CLASSIFIER_COMPARISON.md`), this readout adds a
comparable amount of capacity (3x384 dims) but keeps it spatially
disaggregated, matching the depth grid's successful design.

All other settings (backbone frozen 16 epochs, depth 1x1 pooling,
per-sequence augmentation, optimizer, thresholds) are identical to the
`stable` baseline.

## Results

Each row uses the best `test_passage_macro_f1` epoch for that run.

| Condition | Seed | Epoch | Passage macro-F1 | Direction macro-F1 | Left F1 | Right F1 | Exact accuracy | Turning F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (`last_cls`) | 0 | 14 | 0.6212 | 0.6348 | 0.4490 | 0.5047 | 0.7287 | 0.5806 |
| Regional3 | 0 | 6 | **0.6706** | **0.6485** | **0.4337*** | **0.5625** | **0.7549** | **0.7368** |
| Baseline (`last_cls`) | 1 | 8 | 0.6394 | 0.6363 | 0.4235 | 0.5417 | 0.7418 | 0.6486 |
| Regional3 | 1 | 11 | **0.7048** | **0.7004** | **0.5714** | **0.5833** | **0.7527** | **0.7179** |

\* Seed 0 left F1 is marginally below baseline; every other cell improves
in both seeds individually.

| Two-seed mean | Baseline | Regional3 | Delta |
|---|---:|---:|---:|
| Passage macro-F1 | 0.6303 | **0.6877** | **+0.0573** |
| Direction macro-F1 | 0.6356 | **0.6744** | +0.0389 |
| Left F1 | 0.4363 | **0.5026** | +0.0663 |
| Right F1 | 0.5232 | **0.5729** | +0.0497 |
| Exact accuracy | 0.7352 | **0.7538** | +0.0186 |
| Turning F1 | 0.6146 | **0.7274** | +0.1127 |

The two-seed mean gain (+0.0573 passage macro-F1) is roughly 3.5x larger
than the best previous ablation gain in this experiment series (the depth
4x4 grid, +0.0159). Every metric improves in both seeds, with no
metric-level trade-off, unlike most prior ablations in this series
(imbalance interventions, sequence length, augmentation) which traded one
metric for another or changed sign by seed.

## Combining with the depth 4x4 grid

Since both changes independently help by preserving spatial layout instead
of global-pooling it (RGB regional3 for left/center/right, depth
`depth_pool_size: 4` for a finer spatial grid), the two were combined:
`dino_readout: last_cls_regional3` and `depth_pool_size: 4` together, same
seeds and all other settings unchanged.

| Condition | Seed | Epoch | Passage macro-F1 | Direction macro-F1 | Left F1 | Right F1 | Exact accuracy | Turning F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Regional3 + depth4 | 0 | 15 | 0.6790 | 0.7024 | 0.6019 | 0.5620 | 0.7527 | 0.6087 |
| Regional3 + depth4 | 1 | 15 | 0.6871 | 0.7087 | 0.5437 | 0.6364 | 0.7484 | 0.6222 |

| Two-seed mean | Baseline | Regional3 | Regional3 + depth4 |
|---|---:|---:|---:|
| Passage macro-F1 | 0.6303 | **0.6877** | 0.6830 |
| Direction macro-F1 | 0.6356 | 0.6744 | **0.7056** |
| Left F1 | 0.4363 | 0.5026 | **0.5728** |
| Right F1 | 0.5232 | 0.5729 | **0.5992** |
| Exact accuracy | 0.7352 | **0.7538** | 0.7505 |
| Turning F1 | 0.6146 | **0.7274** | 0.6155 |

Combining the two spatial-layout changes pushes direction macro-F1, left
F1, and right F1 further above regional3 alone. But turning F1 drops back
to roughly the baseline level (0.7274 → 0.6155), which pulls the composite
passage macro-F1 slightly below regional3 alone (0.6877 → 0.6830). This
matches the pattern already seen in
`PASSAGE_DIRECTION_SEQUENCE_COMPARISON.md`: changes that add spatial/
temporal capacity to help the direction heads tend to hurt the turning
head, most likely because the turning positive set is small (96 of 1512
training sequences) and more spatially-detailed features increase
overfitting risk there specifically, or because turning is fundamentally a
motion signal that finer static spatial layout does not help.

## Removing the turning head

The turning/direction trade-off above was resolved at its source: the
`turning` classification task was removed from the model entirely
(`src/corridor_classifier/models.py`, `passage_directions.py`,
`training.py`, `messages.py`). Turning-labeled sequences are now excluded
from the passage-directions dataset at load time
(`CorridorMultiInputDataset`) rather than included with a masked direction
loss and a separate turning head. Downstream consumers that need to know
whether the robot is currently turning are expected to use an existing,
more direct signal instead of a monocular vision classification of it (for
example the navigation stack's own commanded angular velocity) — the
bags used for this dataset do not currently record such a topic, so this
is a forward-looking assumption, not something validated against recorded
data here.

Regional3 + depth4 was retrained without the turning head, same seeds and
all other settings unchanged:

| Condition | Seed | Epoch | Direction macro-F1 | Left F1 | Right F1 | Exact accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Regional3 + depth4 + turning head | 0 | 15 | 0.7024 | 0.6019 | 0.5620 | 0.7527 |
| Regional3 + depth4 + turning head | 1 | 15 | 0.7087 | 0.5437 | 0.6364 | 0.7484 |
| Regional3 + depth4, no turning head | 0 | 8 | 0.6998 | 0.5741 | 0.5909 | 0.7396 |
| Regional3 + depth4, no turning head | 1 | 6 | 0.7260 | 0.6364 | 0.6016 | 0.7681 |

| Two-seed mean | With turning head | No turning head | Delta |
|---|---:|---:|---:|
| Direction macro-F1 | 0.7056 | **0.7129** | +0.0073 |
| Left F1 | 0.5728 | **0.6052** | +0.0324 |
| Right F1 | 0.5992 | 0.5963 | -0.0029 |
| Exact accuracy | 0.7505 | **0.7538** | +0.0033 |

Removing the turning head gives a further small, mostly-consistent gain on
top of the regional3 + depth4 combination, with no metric it clearly hurts
(right F1's -0.0029 is within noise). Since `passage_macro_f1` is now
defined as equal to `direction_macro_f1` (there is no fourth output to
average in), this also simplifies the architecture: no turning classifier
parameters, no turning loss term or positive-weight cap, no turning
threshold to tune.

Combined across the whole series, the direction macro-F1 improvement from
the original `stable` baseline (`last_cls`, 1x1 depth, turning head) to
this final configuration (`last_cls_regional3`, `depth_pool_size: 4`, no
turning head) is **0.6356 → 0.7129, or +0.0773** — roughly 5x the largest
single-change gain found anywhere else in this series.

Artifacts:

```text
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seed0/
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seed1/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seed0/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seed1/
```

## GRU sequence length, revisited without a turning head

`PASSAGE_DIRECTION_SEQUENCE_COMPARISON.md` found that longer GRU history
(7 frames vs 3, both at `frame_stride: 4` / one-second spacing) helped
direction metrics but hurt the turning head, and kept the 3-frame model as
a result. With the turning head removed, that constraint no longer
applies, so the regional3 + depth4, no-turning-head configuration was
retrained with `sequence_length: 7` (same seeds, same 4x4 depth grid,
same regional3 readout, all else unchanged).

| Two-seed mean | seq3 | seq7 | Delta |
|---|---:|---:|---:|
| Direction macro-F1 | **0.7129** | 0.6904 | -0.0225 |
| Left F1 | **0.6052** | 0.5438 | -0.0615 |
| Right F1 | 0.5963 | 0.5916 | -0.0047 |
| Exact accuracy | **0.7538** | 0.7448 | -0.0090 |

Unlike the earlier seq3-vs-seq7 comparison (run with a plain `last_cls`
RGB readout and a 1x1 depth grid), longer history now regresses every
metric. The hypothesis that removing the turning head would unlock the
direction benefit of longer history did not hold here: regional3 and the
4x4 depth grid already give the model a much richer single-frame spatial
representation than the earlier `last_cls` baseline, so the earlier
benefit of aggregating more frames over time was likely compensating for a
weaker per-frame representation rather than being independently useful.
Stacking a longer temporal window on top of the now richer per-frame
features increases input complexity without a matching increase in
training data (about 1400 training sequences either way), and the
regression looks like added overfitting risk rather than a genuine
information deficit. `sequence_length: 3` remains the best setting.

Artifacts:

```text
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seq7_seed0/
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seq7_seed1/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seq7_seed0/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seq7_seed1/
```

## Untouched held-out evaluation

Every comparison above, like every comparison elsewhere in this directory,
selected checkpoints on `test_passage_macro_f1` every epoch against the
same 6 sessions used for the final reported number. That is hyperparameter
selection against the evaluation set, not an untouched estimate of
generalization, and this had been flagged as a standing caveat throughout
the series without being resolved.

To resolve it cheaply (without a full 9-fold leave-one-session-out sweep),
the 6 former "test" sessions were split into a validation subset used only
for per-epoch checkpoint selection (`j, l, m`; 984 images) and a held-out
subset never touched during training or selection (`i, k, n`; 939 images,
222 evaluation sequences at `sequence_length: 3`). Both the baseline
architecture (`last_cls`, 1x1 depth, no turning head) and the final
candidate (`last_cls_regional3`, `depth_pool_size: 4`, no turning head,
`sequence_length: 3`) were retrained from scratch with checkpoints selected
on the validation subset only, then evaluated once on the held-out subset
using `scripts/evaluate_session_holdout.py`.

| Condition | Seed | Held-out direction macro-F1 |
|---|---:|---:|
| Baseline (last_cls, depth 1x1) | 0 | 0.6828 |
| Baseline (last_cls, depth 1x1) | 1 | 0.7272 |
| Final candidate (regional3 + depth4) | 0 | 0.6937 |
| Final candidate (regional3 + depth4) | 1 | 0.6887 |

| Two-seed mean | Baseline | Final candidate | Delta |
|---|---:|---:|---:|
| Held-out direction macro-F1 | **0.7050** | 0.6912 | **-0.0138** |

**The gain does not replicate on untouched data.** Under honest
evaluation, the final candidate is not better than the baseline it was
supposed to improve on — if anything, slightly worse, though within the
range plausibly explained by noise. The entire chain of gains reported
above (regional3 alone +0.0573, + depth4 grid +0.0312 more, + removing the
turning head +0.0073 more, all measured via test-set checkpoint selection)
most likely reflects overfitting the checkpoint/config search to a fixed
6-session, ~1900-image evaluation set across 20+ compared configurations,
not a real improvement in generalization.

A second observation compounds this: the baseline's own seed-to-seed
spread on the held-out set (0.6828 vs 0.7272, a swing of 0.044) is itself
close in magnitude to the "improvement" this whole series was chasing.
With only 3 held-out sessions / 222 evaluation sequences, this dataset is
too small to reliably distinguish these architectures from each other at
all, regardless of evaluation protocol.

This does not mean the regional3 / depth-grid ideas are wrong — the
DINOv2-paper-grounded reasoning behind them (Section 7.4-7.5: dense/
spatial tasks benefit from spatially arranged patch tokens over a global
CLS token) still holds as a design argument. It means this dataset (~6000
training images, 9 sessions total) is not yet large enough to confirm or
reject that reasoning empirically. **Promotion of any of these changes to
the runtime `stable` checkpoint is not recommended until the dataset is
substantially larger** (particularly more held-out sessions, and coverage
of the currently zero-sample `cross_road` class) — see the data-collection
priorities raised earlier in this project.

Artifacts:

```text
runs/experiments/rgb_depth_gru_bags_passage_directions_baseline_no_turning_heldout_seed0/
runs/experiments/rgb_depth_gru_bags_passage_directions_baseline_no_turning_heldout_seed1/
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_heldout_seed0/
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_heldout_seed1/
weights/experiments/rgb_depth_gru_bags_passage_directions_baseline_no_turning_heldout_seed0/
weights/experiments/rgb_depth_gru_bags_passage_directions_baseline_no_turning_heldout_seed1/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_heldout_seed0/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_heldout_seed1/
```

## Decision

**Superseded by the untouched held-out evaluation above — do not promote.**
Every reading in this section was produced by the same test-set
checkpoint-selection protocol that the held-out evaluation later showed
does not predict untouched performance: the combined chain of gains
(regional3 alone +0.0573, + depth4 grid +0.0312, + removing the turning
head +0.0073, totaling +0.0773 over the `stable` baseline on
`test_passage_macro_f1`) did not reproduce when the same two
architectures were retrained with honest validation/held-out session
splits — the final candidate scored 0.0138 *lower* than baseline on
truly unseen sessions (see "Untouched held-out evaluation" above).

None of `last_cls_regional3`, `depth_pool_size: 4`, or removing the
turning head should be promoted to the runtime checkpoint or the `stable`
config on the basis of the numbers in this file. They remain reasonable,
paper-grounded ideas worth revisiting once the dataset is large enough
(more sessions, more held-out data, coverage of missing classes like
`cross_road`) to tell a real improvement apart from checkpoint-selection
noise at this data scale.

Artifacts:

```text
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_seed0/
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_seed1/
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_seed0/
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_seed1/
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seed0/
runs/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_no_turning_seed1/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_seed0/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_seed1/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_seed0/
weights/experiments/rgb_depth_gru_bags_passage_directions_regional3_depth_pool4_seed1/
```
