# Passage-direction imbalance comparison

## Conditions

All conditions use the frozen DINOv2 final-CLS model, 4x4 Depth grid, and
per-sequence augmentation.

- Baseline: direction and turning positive-weight caps are both 2.0.
- Direction pos-weight 4: the left/right weights use their observed ratios
  (3.768 and 3.720), while turning remains capped at 2.0.
- Rare-direction sampler: samples containing an open rare direction receive an
  inverse-square-root sampling factor capped at 2.0. Turning and dead-end
  samples receive the base factor. Loss weights remain capped at 2.0.

## Seed-1 screening

| Condition | Best epoch | Passage macro-F1 | Direction macro-F1 | Left F1 | Right F1 | Exact accuracy | Turning F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 13 | 0.6460 | **0.6443** | **0.4348** | **0.5510** | **0.7352** | 0.6512 |
| Direction pos-weight 4 | 7 | 0.6468 | 0.6344 | 0.4286 | 0.5347 | 0.7199 | 0.6842 |
| Rare-direction sampler | 8 | **0.6525** | 0.6292 | 0.4176 | 0.5253 | 0.7287 | **0.7222** |

Increasing positive weights does not improve either rare direction and is
rejected without a second seed.

## Rare-sampler repeat

| Two-seed mean | Baseline | Rare-direction sampler |
|---|---:|---:|
| Passage macro-F1 | 0.6470 | **0.6531** |
| Direction macro-F1 | **0.6456** | 0.6366 |
| Front F1 | **0.9465** | 0.9434 |
| Left F1 | 0.4348 | **0.4629** |
| Right F1 | **0.5555** | 0.5035 |
| Exact accuracy | **0.7385** | 0.7144 |
| Turning F1 | 0.6512 | **0.7026** |

## Decision

Do not adopt either imbalance intervention. The sampler's higher composite
score comes from turning F1, while direction macro-F1 and exact accuracy fall.
Its left/right trade-off also changes by seed: seed 0 improves left but harms
right, whereas seed 1 harms both. Keep the loss cap at 2.0 and shuffled sampling.

The fixed direction thresholds were intentionally not retuned on the test bags.
Any future asymmetric loss or calibration experiment should tune thresholds on
a separate validation split.
