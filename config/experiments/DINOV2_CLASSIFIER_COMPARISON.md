# DINOv2 classifier comparison

## Design basis

DINOv2's official linear evaluation keeps the backbone frozen and evaluates
the last one or four class tokens, with or without the average-pooled patch
tokens. The implementation follows that readout grid while leaving the existing
RGB + Depth + GRU head unchanged. See the
[DINOv2 paper, Appendix B.3](https://ar5iv.labs.arxiv.org/html/2304.07193)
and the
[official linear evaluation code](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py).

All runs below use the official pretrained DINOv2 ViT-S/14 weights. The DINOv2
backbone is frozen for all 16 epochs. The bags, temporal sampling, augmentation,
loss, optimizer, learning-rate schedule, and decision thresholds are fixed.

## DINOv2 readout comparison (seed 1)

| Readout | Best epoch | Trainable parameters | Passage macro-F1 | Direction macro-F1 | Exact accuracy | Turning F1 |
|---|---:|---:|---:|---:|---:|---:|
| Last CLS | 8 | 639,076 | **0.6394** | **0.6363** | **0.7418** | **0.6486** |
| Last CLS + final patch mean | 12 | 738,148 | 0.6232 | 0.6328 | 0.7330 | 0.5946 |
| Last 4 CLS + final patch mean | 12 | 1,035,364 | 0.6263 | 0.6255 | 0.7374 | 0.6286 |

The larger readouts improve training passage macro-F1 to 0.7290 and 0.7347,
but reduce the test metric. The final CLS alone is retained to avoid adding
capacity that overfits these training bags.

## Spatial Depth comparison

The baseline depth encoder globally averages its final 128 feature maps to 1x1.
The new variant pools them to a 4x4 grid before the 128-dimensional projection.
This preserves coarse left/center/right and near/far layout without changing the
DINOv2 readout or forcing a fixed modality multiplier.

| Seed | Depth grid | Best epoch | Passage macro-F1 | Direction macro-F1 | Exact accuracy | Turning F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1x1 | 14 | 0.6212 | 0.6348 | 0.7287 | 0.5806 |
| 0 | 4x4 | 13 | **0.6469** | **0.6403** | **0.7374** | **0.6667** |
| 1 | 1x1 | 8 | 0.6394 | 0.6363 | 0.7418 | 0.6486 |
| 1 | 4x4 | 10 | **0.6456** | **0.6437** | **0.7462** | **0.6512** |
| Mean | 1x1 | - | 0.6303 | 0.6356 | 0.7352 | 0.6146 |
| Mean | 4x4 | - | **0.6462** | **0.6420** | **0.7418** | **0.6589** |

Per-direction two-seed means also change from 0.9473/0.4363/0.5232 to
0.9475/0.4406/0.5378 for front/left/right. The improvement therefore does not
come only from the already-easy front direction.

## Decision

Use the final CLS DINOv2 readout and the 4x4 Depth grid as the next candidate.
It improves passage macro-F1 in both seeds and gives a two-seed mean gain of
0.0159. The runtime checkpoint is not replaced yet because the training loop
evaluates the test bags every epoch and selects the best checkpoint using test
passage macro-F1. An untouched validation/final-test split is still required for
an unbiased deployment decision.
