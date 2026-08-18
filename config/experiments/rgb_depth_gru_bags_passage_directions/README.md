# RGB + Depth GRU passage directions

既存の9クラスデータセットを、前・左・右の通行可否とturningへ変換して学習する
実験設定です。元画像、depth、`samples.csv`は変更しません。

| source class | front | left | right | direction loss |
| --- | ---: | ---: | ---: | --- |
| straight_road | 1 | 0 | 0 | enabled |
| dead_end | 0 | 0 | 0 | enabled |
| corner_right | 0 | 0 | 1 | enabled |
| corner_left | 0 | 1 | 0 | enabled |
| cross_road | 1 | 1 | 1 | enabled |
| 3_way_right | 1 | 0 | 1 | enabled |
| 3_way_center | 0 | 1 | 1 | enabled |
| 3_way_left | 1 | 1 | 0 | enabled |
| turning | - | - | - | masked |

```bash
roslaunch corridor_classifier train.launch \
  config_dir:=$(rospack find corridor_classifier)/config/experiments/rgb_depth_gru_bags_passage_directions
```

ベストcheckpointは`test_direction_macro_f1`で選択し、次へ保存します。

```text
weights/experiments/rgb_depth_gru_bags_passage_directions/model.pth
```
