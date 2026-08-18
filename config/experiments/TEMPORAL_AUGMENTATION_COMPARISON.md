# Temporal augmentation comparison

## Change

The previous temporal dataset independently sampled ColorJitter order and
factors, grayscale, Gaussian blur, and Depth scale for every GRU frame. The new
mode samples each value once per sequence and applies it to all three frames.
Horizontal flipping and RGB modality dropout were already sequence-wide.

The comparison uses the frozen DINOv2 final-CLS model with a 4x4 Depth grid.
All other settings are fixed.

| Seed | Augmentation | Best epoch | Passage macro-F1 | Direction macro-F1 | Left F1 | Right F1 | Exact accuracy | Turning F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Per-frame | 13 | 0.6469 | 0.6403 | 0.4444 | 0.5283 | 0.7374 | 0.6667 |
| 0 | Per-sequence | 10 | **0.6480** | **0.6469** | 0.4348 | **0.5600** | **0.7418** | 0.6512 |
| 1 | Per-frame | 10 | 0.6456 | 0.6437 | **0.4368** | 0.5474 | **0.7462** | 0.6512 |
| 1 | Per-sequence | 13 | **0.6460** | **0.6443** | 0.4348 | **0.5510** | 0.7352 | 0.6512 |
| Mean | Per-frame | - | 0.6462 | 0.6420 | **0.4406** | 0.5378 | **0.7418** | **0.6589** |
| Mean | Per-sequence | - | **0.6470** | **0.6456** | 0.4348 | **0.5555** | 0.7385 | 0.6512 |

## Decision

Keep per-sequence augmentation because it removes artificial temporal flicker
and improves direction macro-F1 in both seeds. Its passage-score gain is only
0.0008, and it does not improve left F1 or exact accuracy, so it should be
treated as an input-correctness fix rather than a major performance gain.

These runs still select checkpoints on test passage macro-F1 and are controlled
comparisons, not an untouched generalization estimate.
