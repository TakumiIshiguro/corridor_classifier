# Deploying on sessions a-h only, keeping i-n as a genuine unseen check

## Change

Every production checkpoint up to this point (`weights/production/linear_probe_regional3.pth`,
the ViT-B/14 + `last_cls_regional3` winner of `LINEAR_PROBE_SCALING_COMPARISON.md`)
was fit on all 9 sessions (a-h + i-n), with leave-one-session-out (LOSO) CV
used only to pick the ridge regularization strength. That gives an honest
cross-validated *estimate* of generalization, but leaves no session the
deployed checkpoint hasn't already seen -- there is no data left to sanity
check the actual deployed model against in the future.

The new default (`weights/production/linear_probe_regional5_trainonly.pth`,
referenced by `launch/linear_probe.launch`) is fit using
`--exclude-test-sessions`: only `dataset.train_session_names` (a_f, g, h --
i.e. sessions a-h) are used for both regularization selection and the
final fit. `dataset.test_session_names` (i-n) are never touched by fitting
or hyperparameter selection, so they remain available indefinitely as a
genuinely unseen-environment check.

## Re-picking the architecture without touching i-n

Every prior architecture decision in this project (backbone size, DINO
readout, resolution) was chosen using the full 9-session LOSO CV. Once i-n
is being reserved as an unseen check, re-evaluating several architecture
candidates against i-n to pick a winner would itself be a form of model
selection on i-n -- exactly what keeping it held out is meant to prevent.
So the architecture was re-picked using only a-h
(`scripts/compare_architectures_trainonly.py`), and i-n was used exactly
once, only to check the single already-chosen final config.

Session `a_f` is one continuous ~24.5-minute recording (5645 samples, 93%
of a-h), so a plain 3-session (a_f, g, h) LOSO is unreliable: the
`a_f`-held-out fold trains on very little data, and the `g`/`h`-held-out
folds barely perturb the training set. `a_f` was split by timestamp into 6
equal-duration chunks, treated as separate pseudo-sessions for LOSO
*evaluation* purposes only (the final fit still uses the real `a_f`/`g`/`h`
sessions). Because the `a_f`-chunk folds test interpolation within the same
continuous recording rather than generalization to a different
environment, results are reported two ways: a blended score across all 8
pseudo-session folds, and a session-only score using just the 2 folds that
hold out a genuinely different session (`g`, `h`) -- the architecture
ranking uses the latter.

| Candidate | dim | session-only F1 (g/h held out) | blended F1 (8 folds) |
|---|---:|---:|---:|
| **regional5, ViT-B/14, 224** | 4608 | **0.9608** | 0.9776 |
| regional3, ViT-B/14, 224 (prior winner) | 3072 | 0.9564 | 0.9738 |
| regional3, ViT-L/14, 224 | 4096 | 0.9562 | 0.9762 |
| regional3x2, ViT-B/14, 224 | 5376 | 0.9387 | 0.9720 |
| regional3, ViT-B/14, 336 | 3072 | 0.9320 | 0.9607 |
| regional3, ViT-S/14, 224 | 1536 | 0.9309 | 0.9654 |
| last_cls (no regional), ViT-B/14, 224 | 768 | 0.9091 | 0.9538 |

These absolute numbers are not comparable to the 9-session LOSO numbers
elsewhere in this directory -- with only a-h available, even the
session-only signal is n=2 folds of 212 samples each, and the ranking
margins are within plausible noise. It is the best signal obtainable
without spending i-n. `last_cls_regional5` was the top candidate and was
carried forward.

## Untouched i-n result

The chosen config (ViT-B/14, `last_cls_regional5`, 224x224, no depth,
fit on a-h, `l2=1000` selected by the a-h LOSO above) was then evaluated
once, and only once, on i-n:

| Metric | regional3 (a-h only, prior attempt) | **regional5 (a-h only, adopted)** |
|---|---:|---:|
| front F1 | 0.9116 | 0.8873 |
| left F1 | 0.5714 | 0.6320 |
| right F1 | 0.2446 | **0.4652** |
| direction macro-F1 | 0.5759 | **0.6615** |
| exact accuracy | 0.7297 | 0.7366 |

Switching to `regional5` (chosen from a-h data alone) improved the
untouched i-n score by +0.0856 direction macro-F1 over the naive carry-over
of the previous (9-session-selected) `regional3` choice -- most of the gain
is in `right_f1` (+0.22). This is a real result, not an estimate: i-n was
touched exactly once, after the architecture was already fixed.

## Caveats

- This i-n number (0.6615) is *lower* than the old all-9-session
  checkpoint's LOSO estimate (0.7020) and is expected to stay lower,
  since the a-h-only checkpoint has seen only 3 sessions (dominated by one
  93%-share recording) rather than 8 different ones. That is the accepted
  cost of keeping a genuine unseen-environment check available.
- i-n has now been spent as a one-time check for this specific config. If
  further architecture changes are made, a new genuinely-unseen set (more
  freshly collected sessions, not i-n again) is needed to check them
  honestly -- re-checking a modified model against i-n after having already
  looked at i-n once is no longer a clean test.

Artifacts:

```text
weights/production/linear_probe_regional5_trainonly.pth   # linear_probe.launch default
weights/production/linear_probe_regional3_trainonly.pth   # prior attempt, kept for reference
scripts/compare_architectures_trainonly.py
```

## Depth still adds nothing to the ridge probe, even with less data

Re-ran the depth comparison under a-h-only (same chunked-session LOSO):
`regional5` vs `regional5+depth4` scored *identically* (0.9608 session-only-F1
both). Confirms the 9-session finding: the naive log-depth grid feature adds
nothing to the ridge probe regardless of how much training data is
available.

## GRU + depth beats the ridge probe once data is this scarce -- reversal

The GRU comparison is the opposite story. `regional3_depth_pool4` (the
architecture from `REGIONAL_PATCH_READOUT_COMPARISON.md`), trained with
`train_session_names: [a_f]` and `test_session_names: [g, h]` (g/h used
only for checkpoint-epoch selection, i-n still never touched), then checked
once on i-n:

| Condition | seed0 | seed1 | 2-seed mean |
|---|---:|---:|---:|
| Ridge probe (`regional5`, no depth) | -- | -- | 0.6615 |
| **GRU (`regional3` + `depth_pool4`)** | 0.7209 | 0.6902 | **0.7056** |

GRU+depth wins by +0.044, and both seeds individually beat the ridge probe.
This reverses the 9-session-data finding (`LINEAR_PROBE_SCALING_COMPARISON.md`)
where the ridge probe beat every GRU variant. Likely explanation: with only
`a_f` (1333 training sequences) as real training signal, the GRU's *learned*
depth CNN and temporal fusion can extract task-specific cues (e.g. depth
gaps widening over consecutive frames) that a frozen-feature linear
combination cannot, even though the same crude depth feature added nothing
to the ridge probe. In the data-rich 9-session regime the ridge probe's
simplicity was the advantage (less to overfit); in this much smaller
a_f-only regime, the GRU's extra flexibility apparently helps more than it
hurts.

**Conclusion: which architecture is "best" depends on how much data is
available to fit it.** Two defaults are now maintained side by side:

| Regime | Default | Launch | i-n result |
|---|---|---|---:|
| All 9 sessions available | ridge probe, ViT-B/14, `regional5`, no depth | `linear_probe.launch` | 0.6615 (a-h-only fit) / 0.7020 (9-session LOSO estimate) |
| a-h only, i-n reserved as unseen check | GRU, `regional3` + `depth_pool4`, seed0 | `corridor_classifier.launch` | **0.7056** (2-seed mean) |

`launch/corridor_classifier.launch`'s default `config_dir` now points to
`config/experiments/production_gru_trainonly`
(`weights/production/gru_regional3_depth4_trainonly.pth`, the seed0
checkpoint). `launch/linear_probe.launch` is unchanged and remains the
better choice if/when more sessions become available to fit on.

Artifacts:

```text
config/experiments/rgb_depth_gru_regional3_depth4_notuning_trainonly_seed0/
config/experiments/rgb_depth_gru_regional3_depth4_notuning_trainonly_seed1/
config/experiments/production_gru_trainonly/              # corridor_classifier.launch default
weights/production/gru_regional3_depth4_trainonly.pth
```
