# corridor_classifier

ROS Noetic用の、DINOv2による単眼カメラ画像の通路形状分類パッケージです。
カメラ画像を8クラスへ分類し、既存の`intersection_detector`と互換性の
あるメッセージをpublishします。

## Model

- Backbone: DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`)
- Input: RGB, 224 x 224
- Output: 8-class logits
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

入力画像はカメラの視野全体を残すため、アスペクト比を維持せず224 x 224へ
リサイズします。学習時も同じ前処理を使用してください。

## ROS interface

### Subscribed topic

| Topic | Type | Description |
| --- | --- | --- |
| `/camera_center/image_raw` | `sensor_msgs/Image` | Center camera image |

### Published topics

| Topic | Type | Description |
| --- | --- | --- |
| `/passage_type` | `scenario_navigation_msgs/cmd_dir_intersection` | Predicted class |
| `/corridor_classifier/probabilities` | `std_msgs/Float32MultiArray` | Probabilities in configured class order |

`/passage_type`では、`intersection_name`にクラス名、
`intersection_label`に8要素のone-hotベクトルを設定します。分類ノードは
方向指令を生成しないため、`cmd_dir`は常に`[0, 0, 0]`です。

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

学習後の8クラスcheckpointは次の場所に保存され、ROSノードも同じファイルを
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
`intersection_name`は使用しません。全要素0、複数要素1、8要素以外のラベルは
保存しません。

入力方法と使用するbagは`config/dataset.yaml`で指定します。bagファイルは
絶対パス、または`corridor_classifier`パッケージからの相対パスを使用できます。

```yaml
collection:
  source: bag
  bag_path: /absolute/path/to/input.bag
```

bagには`config/topics.yaml`で指定した画像トピックとラベルトピックの両方が
必要です。メッセージをbag記録時刻順に読み、各画像より前に届いた最新の有効な
one-hotラベルを使用します。ラベルが`label_timeout`より古い画像は保存せず、
`sample_dt`間隔で画像を間引きます。画像のROS header stampが有効な場合は、
その時刻を`samples.csv`へ記録します。

設定後は引数なしで起動します。

```bash
roslaunch corridor_classifier create_dataset.launch
```

ライブトピックから収集するときは`source: live`に変更します。その場合、
`bag_path`は使用されません。

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

## Training

学習設定は`config/training.yaml`にあります。
各epochのtrain/test処理ではtqdmにbatch進捗、running loss、accuracy、
learning rateを表示します。

```bash
roslaunch corridor_classifier train.launch
```

または直接実行できます。

```bash
rosrun corridor_classifier train.py \
  --config-dir "$(rospack find corridor_classifier)/config"
```

現在の既定設定では、10 epochすべてbackboneを固定して8クラス分類headだけを
学習します。段階的にunfreezeする場合は`unfreeze_schedule`へepochと
`last_blocks`を追加します。

headとbackboneにはそれぞれ別の学習率を設定します。augmentationは行わず、
収集済み224 x 224画像に`ToTensor`とImageNet正規化だけを適用します。

optimizerとlearning-rate schedulerは`config/training.yaml`で設定します。
既定ではAdamWを使用し、最初の1 epochを初期係数0.1から線形warmupした後、
cosine schedulerで各optimizer stepごとに学習率を減衰します。最終学習率は
head/backboneそれぞれの基準学習率の1%です。`scheduler.name: constant`に
すると、warmup完了後の学習率を一定にできます。

学習結果は次へ保存します。

```text
weights/corridor_classifier.pth
weights/corridor_classifier_final.pth
weights/corridor_classifier.yaml
runs/corridor_classifier/metrics.csv
```

`corridor_classifier.pth`はtest使用時はtest loss、testなしの場合はtrain
lossが最小のcheckpoint、
`corridor_classifier_final.pth`は最終epochのcheckpointです。

## Build

```bash
cd ~/catkin_ws
catkin_make --pkg corridor_classifier
source devel/setup.bash
```

## Run

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
