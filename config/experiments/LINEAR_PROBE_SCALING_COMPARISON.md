# Linear probe backbone size / resolution / regional-grid sweep

## Background

`REGIONAL_PATCH_READOUT_COMPARISON.md` established that a closed-form ridge
regression probe on frozen DINOv2 features (`last_cls_regional3` readout,
ViT-S/14, 224x224, no depth) beats every SGD-trained GRU architecture tried
in this project on an untouched two-fold held-out evaluation, and beats the
original production `stable` checkpoint even when the latter is given an
unfair selection advantage. The production checkpoint was fit on all 9
sessions with the ridge regularization strength (`l2`) chosen by
leave-one-session-out (LOSO) cross-validation, giving a LOSO-estimated
direction macro-F1 of 0.6977.

This sweep asks whether a bigger frozen backbone and/or higher input
resolution pushes that further, now that ridge fitting is cheap enough
(~1-2 minutes per configuration, no SGD) to search broadly.

## New 336x336 dataset

The camera's native resolution is 640x480; the existing dataset was saved
at 224x224 (`dataset/corridor/bags_turning`), discarding real detail. All 9
sessions were re-collected from the original bags at 336x336 (336/14=24, a
clean DINOv2 patch grid) into `dataset/corridor_336`, using the identical
turn-detection settings, giving sample counts that match the 224 dataset
exactly per session (e.g. test/j: 783, test/k: 604, ...).

## Backbone x resolution grid

All runs: `last_cls_regional3` readout, no depth, ridge fit on all 9
sessions, `l2` chosen by LOSO CV (candidates 0.1 to 1e5).

| Backbone | Params | 224 (existing data) | 336 (new data) |
|---|---:|---:|---:|
| ViT-S/14 | 22M | 0.6977 | 0.6915 |
| **ViT-B/14** | 86M | **0.7020** | 0.6835 |
| ViT-L/14 | 303M | 0.6931 | 0.6744 |

Two consistent findings, both somewhat counter-intuitive:

1. **336 is worse than 224 for every backbone size.** More pixels did not
   help despite the camera's native resolution being higher than 224.
2. **Bigger is not monotonically better.** ViT-B/14 beats both ViT-S/14 and
   ViT-L/14; ViT-L/14 is the worst of the three at both resolutions.

The likely explanation for both is the same: regional3 features scale
linearly with backbone width (4 x num_features) and this project's dataset
is small (~7500 training sequences after excluding turning frames). ViT-L's
4096-dim regional3 features and/or the finer patch grid from 336x336 input
increase the ridge regression's effective input dimensionality beyond what
~7500 samples can support well, even with cross-validated regularization.
ViT-B/14 (3072-dim features) appears to be the capacity/data sweet spot for
this dataset size.

## Regional grid granularity (ViT-B/14, 224)

Tested whether finer spatial partitioning helps further, given the
DINOv2-paper-grounded reasoning that motivated `last_cls_regional3` in the
first place. Added `last_cls_regional5` (5 columns) and
`last_cls_regional3x2` (3 columns x 2 rows, a 2D grid) readouts to
`models.py`'s `DINO_READOUTS` / `regional_patch_features` (now generalized
to accept `rows` in addition to `columns`).

| Readout | Direction macro-F1 (LOSO) |
|---|---:|
| **regional3** (1x3) | **0.7020** |
| regional5 (1x5) | 0.6991 |
| regional3x2 (2x3) | 0.6957 |

Same pattern again: finer partitioning adds dimensions without adding
useful signal at this data scale, and slightly hurts. `regional3` remains
the best granularity found.

## Decision

Production checkpoint updated to **ViT-B/14, 224x224, `last_cls_regional3`,
no depth**, fit on all 9 sessions, `l2=30000` selected by LOSO CV. LOSO
direction macro-F1 = **0.7020** (previously 0.6977 with ViT-S/14) — a real
but modest further gain of +0.0043 on top of the much larger gains already
captured by the regional3 idea itself and by moving off the GRU
architecture entirely.

DINOv2 ViT-B/14 pretrained weights were downloaded once via timm and pinned
locally to `weights/dinov2_vitb14_pretrain.pth` (verified bit-identical
forward-pass output against the timm hub download before use) so that
inference does not depend on network access or on timm's hub cache
matching what was used at fit time.

Diminishing/negative returns from scaling further (backbone size,
resolution, and regional granularity all plateaued or regressed past this
point) suggest the next real gains have to come from more/better data
(more sessions, coverage of the still-empty `cross_road` class) rather than
from more model capacity — consistent with every other finding in this
project's experiment series.

Artifacts:

```text
dataset/corridor_336/                                    # new 336x336 re-collection, all 9 sessions
weights/dinov2_vitb14_pretrain.pth                        # pinned ViT-B/14 backbone weights
weights/experiments/linear_probe_vits14_336/probe.pth
weights/experiments/linear_probe_vitb14_224/probe.pth
weights/experiments/linear_probe_vitb14_336/probe.pth
weights/experiments/linear_probe_vitl14_224/probe.pth
weights/experiments/linear_probe_vitl14_336/probe.pth
weights/experiments/linear_probe_vitb14_224_regional5/probe.pth
weights/experiments/linear_probe_vitb14_224_regional3x2/probe.pth
weights/production/linear_probe_regional3.pth             # final: ViT-B/14, 224, regional3
```
