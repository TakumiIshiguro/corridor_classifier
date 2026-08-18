# corridor_classifier

ROS Noetic用の、DINOv2による単眼カメラ画像の通路形状分類パッケージです。
カメラ画像を8種類の通路形状と旋回中の計9クラスへ分類し、既存の`intersection_detector`と互換性の
あるメッセージをpublishします。

## Model

- Backbone: DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`)
- Input: RGB and optional UniDepth metric depth, 224 x 224
- Output: configurable 9-class logits or passage-direction/turning logits
- Default inference rate: 4 Hz

クラスの順序は次のとおりです。

1. `straight_road`
2. `dead_end`
3. `corner_right`
4. `corner_left`
5. `cross_road`
6. `3_way_right`
7. `3_way_center`
8. `3_way_left`
9. `turning`

`model.output_mode`は次の2方式を選択できます。

- `class`: 従来の9クラス排他的分類
- `passage_directions`: `front`、`left`、`right`の独立3 logitsと、独立した
  `turning` logit

`passage_directions`では既存9クラスラベルを読み込み時に3方向へ変換します。
`turning`サンプルでは通路方向が曖昧なため方向lossをmaskし、turning headだけを
学習します。データセット自体の作り直しは不要です。

入力画像はカメラの視野全体を残すため、アスペクト比を維持せず224 x 224へ
リサイズします。学習時も同じ前処理を使用してください。

`config/model.yaml`の`model.architecture`だけで学習・推論モデルを切り替えます。

| architecture | Input | Network |
| --- | --- | --- |
| `rgb` | Current RGB | DINOv2 + linear head |
| `rgb_gru` | Five RGB frames | DINOv2 + GRU |
| `rgb_depth` | Current RGB and depth | DINOv2 + depth CNN + fusion head |
| `rgb_depth_gru` | Five RGB/depth pairs | DINOv2 + depth CNN + GRU |

学習・推論launchにarchitectureの上書き引数はありません。

## ROS interface

### Subscribed topic

| Topic | Type | Description |
| --- | --- | --- |
| `/camera_center/image_raw` | `sensor_msgs/Image` | Center camera image |
| `/unidepth/depth` | `sensor_msgs/Image` (`32FC1`) | Metric depth; subscribed only by depth architectures |

### Published topics

| Topic | Type | Description |
| --- | --- | --- |
| `/passage_type` | `scenario_navigation_msgs/cmd_dir_intersection` | Predicted class |
| `/corridor_classifier/probabilities` | `std_msgs/Float32MultiArray` | Class probabilities, or `[front, left, right, turning]` |

`/passage_type`では、`intersection_name`にクラス名、
`intersection_label`は既存メッセージとの互換性のため8要素のままです。通路形状
クラスでは従来のone-hotを設定し、`turning`では全要素を0にして
`intersection_name=turning`を設定します。分類ノードは
方向指令を生成しないため、`cmd_dir`は常に`[0, 0, 0]`です。
`passage_directions`では閾値処理後の3方向から従来の8形状名とone-hotを復元する
ため、既存の`/passage_type`利用側との互換性を維持します。turning判定を優先し、
turning中は従来どおり8要素one-hotをすべて0にします。

トピック名は`config/topics.yaml`で変更できます。

## Requirements

- ROS Noetic
- Python 3
- PyTorch
- torchvision
- `timm >= 1.0.0`
- Pillow
- PyYAML
- `cv_bridge`
- tqdm

現在のワークスペースにある`timm 1.0.19`で動作するモデル名を使用しています。

## Checkpoint

事前学習済みDINOv2 ViT-S/14重みを次の場所へ配置します。

```text
weights/dinov2_vits14_pretrain.pth
```

既定設定はレジスタなしモデルを使用します。`reg4`重みは使用しません。

学習後の9クラスcheckpointは次の場所に保存され、ROSノードも同じファイルを
読み込みます。

```text
weights/corridor_classifier.pth
```

対応する保存形式は次のいずれかです。

```python
# Plain state dictionary
torch.save(model.state_dict(), path)

# Recommended format
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    },
    path,
)
```

checkpointのモデル名、入力サイズ、分類ヘッド、クラス順序は
`config/model.yaml`と一致させてください。異なるクラス順序がcheckpointに
記録されている場合、ノードは誤った分類結果をpublishせず起動時に停止します。

## Dataset collection

`waypoint_navigator_with_direction_intersection_detailed`がpublishする
`/cmd_dir_intersection`の8要素one-hotラベルとカメラ画像を保存します。
元メッセージの`intersection_name`は使用しません。全要素0、複数要素1、8要素
以外の元ラベルは保存しません。bag収集では`/mcl_pose`からyaw角速度を求め、
旋回中の画像だけ9番目の`turning`へ置換します。

入力方法と使用するbagは`config/dataset.yaml`で指定します。bagファイルは
絶対パス、または`corridor_classifier`パッケージからの相対パスを使用できます。

```yaml
collection:
  source: bag
  bag_path: /absolute/path/to/input.bag
```

bagには`config/topics.yaml`で指定した画像トピックとラベルトピックの両方が
必要です。旋回判定を有効にした場合は`turn_detection.pose_topic`も必要です。
メッセージをbag記録時刻順に読み、各画像より前に届いた最新の有効な
one-hotラベルを使用します。ラベルが`label_timeout`より古い画像は保存せず、
`sample_dt`間隔で画像を間引きます。画像のROS header stampが有効な場合は、
その時刻を`samples.csv`へ記録します。

旋回判定の既定値は、中央1秒窓のyaw角速度が`0.20 rad/s`以上で0.5秒以上
継続する区間です。前後0.25秒も`turning`へ含めます。

```yaml
turn_detection:
  enabled: true
  source_num_classes: 8
  class_name: turning
  pose_topic: /mcl_pose
  angular_speed_threshold_rad_s: 0.20
  window_seconds: 1.0
  minimum_duration_seconds: 0.5
  padding_seconds: 0.25
  post_turn_next_label_max_seconds: 6.0
  turning_gap_bridge_max_seconds: 1.5
```

`A -> turning -> A -> B`のように旋回後も旧ラベル`A`が短時間残る場合、
後段の`A`が6秒以内なら次ラベル`B`へ置換します。旋回終了後の直線画像を
`turning`へ延長せず、次の通路形状へ揃えるためのbag専用後処理です。
また、`turning -> A -> turning`の間にある`A`が1.5秒以内なら、旋回中の
一時的な直進とみなして`turning`へ統合します。

設定後は引数なしで起動します。

```bash
roslaunch corridor_classifier create_dataset.launch
```

ライブトピックから収集するときは`source: live`に変更します。その場合、
`bag_path`は使用されません。中央時間窓を必要とする旋回判定はbag専用なので、
ライブ収集では`turn_detection.enabled: false`にしてください。

画像は収集時に視野全体を224 x 224へリサイズして保存します。保存間隔、
ラベルtimeout、出力先は`config/dataset.yaml`で設定します。

```text
dataset/corridor/train/session_YYYYMMDD_HHMMSS/
├── images/
│   ├── 000000.jpg
│   └── ...
├── samples.csv
└── metadata.yaml
```

収集を終了するときはCtrl-Cを使用します。testデータを収集する場合は
次のように起動します。

```bash
roslaunch corridor_classifier create_dataset.launch \
  dataset_type_override:=test
```

trainとtestは`config/training.yaml`の`train_data_dir`と`test_data_dir`で
別々に指定します。testを使用する場合は`training.use_test: true`にします。
`use_test: false`ではtestデータを読み込まず、trainデータだけで学習します。

既存の画像とdepthを再利用して旋回ラベル版を作る場合は、次を実行します。
出力側の`images/`と`depth/`は元データへの相対シンボリックリンクになるため、
画像や深度を複製しません。

```bash
rosrun corridor_classifier relabel_turning.py \
  --source-dir dataset/corridor/bags_all_train/train \
  --output-dir dataset/corridor/bags_turning/train \
  --config-dir config
```

Depthモデルを学習する前に、各sessionの`metadata.yaml`に記録された元bagを
UniDepthで再処理し、対応するメートル深度を追加します。

```bash
roslaunch corridor_classifier add_depth_to_dataset.launch
```

対象split、session名、UniDepth設定、stamp許容誤差は
`config/dataset.yaml`の`depth_generation`で指定します。Depthは224 x 224の
`float16` NumPy配列として`depth/`へ保存され、`samples.csv`へ
`depth_filename`列が追加されます。

## Training

学習設定は`config/training.yaml`にあります。
各epochのtrain/test処理ではtqdmにbatch進捗、running loss、accuracy、
learning rateを表示します。

```bash
roslaunch corridor_classifier train.launch
```

3方向multi-label + turning headを新しいbagデータセットで学習する設定は次です。

```bash
roslaunch corridor_classifier train.launch \
  config_dir:=$(rospack find corridor_classifier)/config/experiments/rgb_depth_gru_bags_passage_directions
```

この設定はDINOv2公式事前学習済み重みから開始し、3フレームを1秒間隔で使用
します。epoch 1〜3はDINOv2をfreezeし、epoch 4以降は最後のTransformer
Block 2層と最終Normだけをunfreezeします。

新しいbagデータで採用している安定化設定は次です。

```bash
roslaunch corridor_classifier train.launch \
  config_dir:=$(rospack find corridor_classifier)/config/experiments/rgb_depth_gru_bags_passage_directions_stable
```

この設定はseed 1を使用し、DINOv2 backboneを全epoch固定します。head学習率
`2e-5`、2 epochの
warmup、正例重み上限`2.0`を使います。判定閾値は`front=0.25`、
`left=0.70`、`right=0.70`、`turning=0.60`です。checkpointは方向3出力と
turningを同じ重みで平均した`test_passage_macro_f1`が最大のepochを保存します。
閾値を再評価する場合は次を実行します。

```bash
python3 scripts/tune_passage_thresholds.py \
  --config-dir config/experiments/rgb_depth_gru_bags_passage_directions_stable
```

または直接実行できます。

```bash
rosrun corridor_classifier train.py \
  --config-dir "$(rospack find corridor_classifier)/config"
```

現在の既定設定では、10 epochすべてbackboneを固定して9クラス分類headだけを
学習します。段階的にunfreezeする場合は`unfreeze_schedule`へepochと
`last_blocks`を追加します。

headとbackboneにはそれぞれ別の学習率を設定します。augmentationの有無と強度は
`training.augmentation`で指定します。

optimizerとlearning-rate schedulerは`config/training.yaml`で設定します。
既定ではAdamWを使用し、最初の1 epochを初期係数0.1から線形warmupした後、
cosine schedulerで各optimizer stepごとに学習率を減衰します。最終学習率は
head/backboneそれぞれの基準学習率の1%です。`scheduler.name: constant`に
すると、warmup完了後の学習率を一定にできます。

学習結果は選択したarchitectureごとに保存します。

```text
weights/corridor_classifier.pth                       # rgb
weights/corridor_classifier_rgb_gru.pth
weights/corridor_classifier_rgb_depth.pth
weights/corridor_classifier_rgb_depth_gru.pth
runs/corridor_classifier/<architecture>/metrics.csv
```

各checkpointは`training.checkpoint_metric`で指定した指標が最良のモデルです。
3方向方式では方向別F1、3方向macro-F1、3方向完全一致率、turning F1/accuracy、
および4出力の`passage_macro_f1`を記録します。最終epochは対応する
`*_final.pth`へ保存します。

## Build

```bash
cd ~/catkin_ws
catkin_make --pkg corridor_classifier
source devel/setup.bash
```

## Run

今回採用したseed 1のRGB+Depth+GRUモデルは、UniDepthを含む専用launchで
起動します。`/camera_center/image_raw`を受け取り、4 Hzで
`/passage_type`と`/corridor_classifier/probabilities`をpublishします。

```bash
roslaunch corridor_classifier passage_directions.launch
```

すでに別ノードが`/unidepth/depth`をpublishしている場合は、深度推定の重複起動を
避けます。

```bash
roslaunch corridor_classifier passage_directions.launch \
  start_depth_estimator:=false
```

従来の9クラス分類設定や任意のconfigを使用する場合は、汎用launchを使います。

```bash
roslaunch corridor_classifier corridor_classifier.launch
```

## Feature visualization

通常の推論ノードとは分離して、データセット内の数枚に対するDINOv2 patch特徴を
確認できます。対象データと枚数は
`config/feature_visualization.yaml`で設定します。

```bash
roslaunch corridor_classifier visualize_features.launch
```

指定枚数を一度だけ推論し、入力画像とRGB PCA特徴マップを並べたPNGを
`output_dir/YYYYMMDD_HHMMSS_microseconds/seed_<値>/`へ保存して終了します。
ROS画像のpublishやリアルタイム表示は行わないため、通常の推論ノードには
追加負荷がありません。
各PNGは、入力画像、通路分類でfine-tuning済みのDINOv2 ViT-S/14特徴、
ImageNet教師あり事前学習済みのViT-S/16特徴、ResNet-18特徴を横に並べます。
ImageNetモデルは`feature_visualization.yaml`の`imagenet_vit`と
`imagenet_resnet`で指定します。ResNetは最終畳み込み層の空間特徴をPCA表示
します。`weights_path`が空の場合、初回実行時に`timm`が事前学習済み重みを
ダウンロードし、以降はキャッシュを使用します。
各ラベルから`min_images_per_class`枚を優先して選択します。必要枚数未満の
ラベルはROS警告へ表示されます。
`seeds`へ複数の整数を指定すると、ラベルごとの最低枚数を維持したまま
seedごとに異なる画像をランダム選択し、`seed_<値>/`へ分けて保存します。

checkpoint、device、推論周期はlaunch引数から一時的に上書きできます。

```bash
roslaunch corridor_classifier corridor_classifier.launch \
  checkpoint_path_override:=/absolute/path/to/model.pth \
  device_override:=cuda \
  inference_rate_override:=8.0
```

CUDAが利用できない環境では`device: auto`がCPUを選択します。`use_fp16`は
CUDA使用時のみ有効になります。

## Test

```bash
cd ~/catkin_ws/src/corridor_classifier
pytest -q
```

## License

このパッケージ独自のコードはMIT Licenseです。DINOv2および`timm`は
Apache License 2.0です。詳細は`THIRD_PARTY_NOTICES.md`を参照してください。
