# Passage-direction GRU sequence-length comparison

## Conditions

The corrected-label, frozen-backbone RGB-depth-GRU model was compared with
three, five, and seven GRU frames. All inputs are sampled from the 4 Hz stream
with `frame_stride: 4`, so adjacent GRU inputs are one second apart. Optimizer,
augmentation, thresholds, and all other training settings were kept fixed.

The `test` split is used for checkpoint selection in these experiments. These
are controlled selection results, not an untouched estimate of generalization.

## Seed-1 results

Each row uses the best `test_passage_macro_f1` epoch for that run.

| GRU frames | History | Required 4 Hz messages | Epoch | Train sequences | Passage macro-F1 | Direction macro-F1 | Left F1 | Right F1 | Exact match | Turning F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 s | 9 | 8 | 1512 | 0.6394 | 0.6363 | 0.4235 | **0.5417** | 0.7418 | **0.6486** |
| 5 | 4 s | 17 | 10 | 1506 | 0.6333 | 0.6444 | 0.4500 | 0.5376 | 0.7438 | 0.6000 |
| 7 | 6 s | 25 | 9 | 1500 | **0.6455** | **0.6555** | **0.4789** | 0.5412 | **0.7575** | 0.6154 |

## Seed-0/1 repeat comparison

Because the seven-frame improvement was small, the three- and seven-frame
conditions were also compared across seeds 0 and 1.

| Two-seed mean | 3 frames | 7 frames |
| --- | ---: | ---: |
| Passage macro-F1 | **0.6303** | 0.6270 |
| Direction macro-F1 | 0.6356 | **0.6425** |
| Exact match | 0.7352 | **0.7494** |
| Turning F1 | **0.6146** | 0.5804 |

Longer history consistently helps the direction heads and exact-match score,
but hurts the turning head enough that the composite passage score does not
improve across seeds. It also increases startup context from two to six
seconds. The adopted single-GRU runtime therefore remains the three-frame
seed-1 checkpoint.

The result suggests a multi-timescale follow-up: use a seven-frame direction
GRU and a short three-frame turning GRU, or derive turning from odometry. This
would retain the direction benefit without forcing the motion-state head to
consume stale six-second context.
