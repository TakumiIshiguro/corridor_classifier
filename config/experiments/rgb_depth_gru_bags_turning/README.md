# RGB + Depth GRU with turning class

`/mcl_pose`から検出した旋回区間を9番目の`turning`クラスへ置換した、
4 Hzのbagデータセット用設定です。RGBとUniDepthのメートル深度は、元bagから
新規生成しています。

旋回判定は中央1秒窓のyaw角速度`0.20 rad/s`を使用します。直線区間で確認
された`0.120`〜`0.187 rad/s`のMCL yaw揺れを除外するため、当初の
`0.12 rad/s`から引き上げています。
`A -> turning -> A -> B`で旋回後に旧ラベル`A`が6秒以内だけ残る場合は、
直線画像を`turning`へ延長せず、次ラベル`B`へ置換します。
`turning -> A -> turning`の間にある1.5秒以内の`A`は、旋回中の一時的な
直進として`turning`へ統合します。

## Dataset split

- train: `a-f.bag` (`a_f`), `g.bag`, `h.bag`
- test: `i.bag` - `n.bag`
- root: `dataset/corridor/bags_turning`
- RGB/depth size: 224 x 224
- total: train 6,069 samples, test 1,923 samples
- GRU sequence: 3 frames, stride 4 at 4 Hz (0 s, 1 s, 2 s)
- available sequences: train 1,512, test 471

Raw-frame class counts are shown below.

| class | train | test |
| --- | ---: | ---: |
| straight_road | 3,402 | 1,352 |
| dead_end | 46 | 94 |
| corner_right | 298 | 37 |
| corner_left | 275 | 38 |
| cross_road | 0 | 0 |
| 3_way_right | 640 | 167 |
| 3_way_center | 368 | 56 |
| 3_way_left | 648 | 124 |
| turning | 392 | 55 |

`cross_road`は元bagに有効サンプルがないため、このデータだけでは学習・評価
できません。9クラス分類として使用する場合は、将来このクラスを含むbagを追加
する必要があります。

## Training

```bash
roslaunch corridor_classifier train.launch \
  config_dir:=$(rospack find corridor_classifier)/config/experiments/rgb_depth_gru_bags_turning
```
