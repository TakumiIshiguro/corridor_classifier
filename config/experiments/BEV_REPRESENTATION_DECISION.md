# BEV representation

Depth reaches the classifier as a downsampled depth map. The alternative
here projects the same UniDepth point cloud into a bird's-eye occupancy
grid. On the unseen area it beats RGB+depth, but only after a defect that
invalidated the first ten experiments was fixed.

## The defect

`bev_pool_size: 1` put `AdaptiveAvgPool2d((1, 1))` at the end of
`BevEncoder`, averaging the whole 50x70 grid into a single value per
channel. The one thing a BEV says about a junction -- *where* the opening
is -- was destroyed before the fusion layer saw it, which is exactly the
left/right information the earlier runs kept failing to gain. Depth never
had this problem: `depth_pool_size` was 4 all along.

Measured against an RGB-only control at 7 epochs on the unseen area (i-n):

| input                | macro  | contribution |
|----------------------|--------|--------------|
| RGB only             | 0.6637 | --           |
| RGB + BEV, pool 1    | 0.6691 | +0.005       |
| RGB + depth          | 0.7539 | +0.090       |
| RGB + BEV, pool 4    | 0.7774 | +0.114       |

The fusion layer confirmed the diagnosis independently: the BEV slice held
4.19% of the weight energy for 4.0% of the input width, per-unit weights
1.025x RGB's, LayerNorm gains still at their 1.00 initialisation. The
branch had not been learned away -- it had nothing left to say.

Every BEV result recorded before this fix is void, including the earlier
conclusion that BEV was not worth adopting.

## Result after the fix

Two seeds, 7 epochs, final checkpoint, flat 0.5 thresholds, sim training
(a-h) evaluated on the unseen area (i-n):

| input                    | macro (s0/s1)   | mean   | front | left  | right |
|--------------------------|-----------------|--------|-------|-------|-------|
| RGB + depth              | 0.7539 / 0.7433 | 0.7486 | 0.905 | 0.580 | 0.761 |
| RGB + wall BEV, pool 4   | 0.7774 / 0.7733 | 0.7754 | 0.934 | 0.642 | 0.750 |

The seed ranges do not overlap. BEV gains on front (+0.029) and left
(+0.062) and gives up a little on right (-0.011).

`bev_feature_dim` was raised 64 -> 128 at the same time, matching
`depth_feature_dim`. A 2x2 ablation separates the two changes, and the
pooling is doing nearly all of the work:

| pool | dim | macro  | left  |
|------|-----|--------|-------|
| 1    | 64  | 0.6769 | 0.466 |
| 1    | 128 | 0.6708 | 0.446 |
| 4    | 64  | 0.7617 | 0.603 |
| 4    | 128 | 0.7774 | 0.655 |

Pooling 1 -> 4 is worth +0.085 at dim 64 and +0.107 at dim 128. Widening
64 -> 128 is worth -0.006 at pool 1 and +0.016 at pool 4: with the grid
averaged to a point there is nothing for the extra width to carry, and a
wider projection of the same global summary is if anything slightly worse.
Width only pays once the spatial layout survives. Against the RGB-only
control at 0.6637, a pool-1 BEV branch is worth +0.013 at best.

Past 4 the macro plateaus -- the conv stack leaves a 13x18 map, so 12 is
already near native resolution:

| pool | macro  | front | left  | right | train | proj params |
|------|--------|-------|-------|-------|-------|-------------|
| 1    | 0.6708 | 0.915 | 0.446 | 0.651 | 0.887 | 0.02M       |
| 4    | 0.7774 | 0.933 | 0.655 | 0.744 | 0.922 | 0.26M       |
| 6    | 0.7650 | 0.935 | 0.658 | 0.702 | 0.922 | 0.59M       |
| 8    | 0.7762 | 0.931 | 0.671 | 0.726 | 0.923 | 1.05M       |
| 12   | 0.7702 | 0.930 | 0.698 | 0.682 | 0.942 | 2.36M       |

The 0.0124 spread from 4 up is inside the seed noise measured elsewhere
(0.004-0.011). The components are not flat though: left climbs
monotonically with resolution (0.655 -> 0.698) while right falls
(0.744 -> 0.682), and the two cancel. Left is evidently still
resolution-limited at pool 12; whether right's decline is overfitting
(train F1 is highest there, 0.942) or noise is unresolved on one seed.
Pool 4 is kept for the best macro at the fewest parameters.

Depth and BEV are not complementary -- keeping both is worse than BEV
alone (`cc_bevwall_depth_pool4`):

| input                      | macro  | front | left  | right |
|----------------------------|--------|-------|-------|-------|
| RGB + depth                | 0.7486 | 0.905 | 0.580 | 0.761 |
| RGB + depth + wall BEV     | 0.7595 | 0.931 | 0.587 | 0.761 |
| RGB + wall BEV             | 0.7754 | 0.934 | 0.642 | 0.750 |

Adding depth pulls left back from 0.642 to 0.587, its RGB+depth level,
while right stays at depth's 0.761 either way. The best configuration
drops the depth branch entirely. UniDepth still runs -- the BEV grid is
built from its point cloud -- but the depth-map CNN over 224x224 is
replaced by the BEV CNN over 50x70.

## Floor band vs wall band

The wall band (`z` in [0.05, 1.6]) encodes a side opening as a gap in the
wall -- an absence, in the part of the image where the depth estimate is
sparsest, which is also what dropout and occlusion produce. The floor band
(`z` in [-0.30, -0.05], median floor height -0.106 m) turns the same
opening into observed floor extending sideways, so faking it takes a false
measurement rather than a missing one. Floor is also much denser: 40% of
all points against 27%, and 751-2043 occupied cells per frame against
112-332. See `runs/bev_inspection/floor_vs_wall.png` and
`floor_bev_class_mean.png`.

It does not work. With the pooling fixed and everything else equal:

| band  | macro  | front     | left  | right |
|-------|--------|-----------|-------|-------|
| wall  | 0.7774 | 0.933     | 0.655 | 0.744 |
| floor | 0.6965 | **0.944** | 0.518 | 0.628 |

Floor wins `front` outright -- it is the best `front` score measured from
any input -- and loses badly on left and right. The class means show why:
`3_way_left` and `3_way_right` barely differ from `straight_road` in floor
extent, because a three-way opening is a hole in a wall rather than a
widening of the floor, whereas `corner_left` and `corner_right` show the
bulge clearly.

## A second trap

`bev_floor_filename` was added at the head of the manifest-column priority
list, so `cc_bev5m_ep7` -- a wall config -- silently read the floor grid
instead, and the two scored within 0.0016 of each other. Sessions
accumulate one column per BEV experiment, so the column is now named
explicitly: set `model.bev_manifest_column` (`bev5m_filename`,
`bev_floor_filename`, ...). `BEV_GRID_COLUMNS` in `dataset.py` only
supplies the default when nothing is named.

## Not yet adopted

The ROS path does not implement BEV: `CorridorPredictor.predict()` calls
`encode_frames(rgb, depth)`, so a `use_bev: true` checkpoint raises
`ValueError: bev input is required by this architecture`. The
point-cloud-to-grid step would also have to run every frame at 4 Hz.
