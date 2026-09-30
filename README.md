# NeWRF → visual VAE latent alignment

RF-3DGSの受信機位置で学習できたNeWRFを使う、案②の試作です。
カメラ視野内のRF特徴マップをCNNで読み、対応する可視画像のVAE潜在平均 μ を予測します。

## 試作の構成

1. 学習済みNeWRFを固定して、画像の画素配置に沿うレイを問い合わせる。
2. 各サンプル点で `model.output` の入力を取得する。方向条件付き分岐の最終I/Q出力の直前の特徴で、標準構成では64次元。
3. 学習時の合成器が返す重み `w` で `sum(w * h)` を計算し、`sum(w)` を追加する。標準設定は `[65, 32, 48]`。
4. 同じ位置・姿勢・画角でレンダリングしたGS画像を、固定したVAEで符号化して μ を得る。
5. RF特徴マップ → 小型CNN → 予測 μ を学習する。損失は元のVAE座標における平均二乗誤差 `MSE(pred_mu, target_mu)`。
6. 固定VAEのデコーダで画像を復元する。

学習対象はCNNだけです。NeWRFとVAEの重みは変えません。VAEのサンプリングした z は使いません。
既存の `train_gray_vae2.py` / `train_projection_head2.py` と同じ `GrayVectorVAE` を使うため、
この試作の復元画像はグレースケールです。潜在次元はチェックポイントの設定から読みます。

## まず用意するもの

| 入力 | 内容 |
|---|---|
| NeWRFソース | 成功した学習時の `models.py`, `encoders.py`, `samplers.py`, `synthesizers.py` |
| NeWRFチェックポイントとYAML | RF-3DGS受信機位置で学習できたモデル、その学習設定 |
| カメラと教師画像 | 同じ受信機位置・姿勢・画角のGS画像、カメラの外部・内部パラメータ |
| 座標変換 | GSのワールド座標 → NeWRF座標の4×4変換 |
| VAEチェックポイント | 既存の学習済みGrayVectorVAE。設定が保存されていなければ設定JSONも必要 |

実験用チェックポイントや画像はこのリポジトリには含みません。
各パスは、Pythonを動かす同じ環境・コンテナから読める必要があります。
既存の `docker_run.sh` は本リポジトリだけを `/workspace` にマウントします。
別リポジトリのNeWRFや外部データを使う場合、そのディレクトリもコンテナにマウントしてください。

```bash
python -m pip install -r requirements-alignment.txt
```

PyTorchは現在使用しているCUDA環境に合うものを利用してください。

## 1. 画像とカメラの対応を作る

`configs/views.example.json` は形式の例であり、実験の座標や画像パスではありません。
`rf_from_world` の単位変換・回転・並進を含め、実データの値に置き換えてください。
座標が一致していると確認できた場合にだけ単位行列を指定します。

各視点には次が必要です。

- `view_id`: 画像とRF特徴を結び付ける一意なID。
- `receiver_id`: 同じ受信機位置の全視点で共通のID。
- `image`: その視点のGS画像。相対パスはmanifestのある場所を基準に解決。
- `camera_to_world`: カメラ座標 → GSワールド座標。world-to-camera行列ではない。
- `intrinsics`: 画像サイズと `fx, fy, cx, cy`。
- `split`: 任意。指定する場合は全視点に `train` / `val` / `test` を指定。

`camera_convention` はOpenCV（右・下・前が+X,+Y,+Z）または
OpenGL（右・上・後ろが+X,+Y,+Z）を明示します。
内部パラメータは画像の左上境界を原点とし、最初の画素中心を `(0.5, 0.5)` とする形式です。
整数座標を画素中心とする較正値を使う場合は、主点の座標原点を合わせてください。
カメラの中心をNeWRFの受信機位置に一致させます。

Blender/NeRF形式の `transforms_train.json` がある場合は変換できます。
`rf_from_world.json` の内容は4×4行列のJSON配列です。

```bash
python -m rrf_alignment.prepare_views \
  --transforms /path/to/transforms_train.json \
  --rf-from-world /path/to/rf_from_world.json \
  --image-map /path/to/gs_image_map.json \
  --output /workspace/Temp/newrf_alignment/views.json
```

`--image-map` は元の `file_path` → 実際のGSレンダリング画像のパスを対応付けるJSONです。
相対パスはこのJSONの場所が基準です。画像名やソート順だけでは対応を推定しません。

```json
{
  "./train/rx000_view000": "/path/to/gs/renders/00000.png",
  "./train/rx001_view000": "/path/to/gs/renders/00001.png"
}
```

`--image-map` を省略した場合は `file_path` の元画像を使います。
GSの出力画像に置き換える際は、そのレンダリングに使ったカメラとの対応を確認してください。
変換対象はOpenGLのcamera-to-worldを持つBlender形式です。
COLMAPや3DGSの `cameras.json` は別形式なので、そのまま渡せません。
`camera_angle_x` だけがある場合は、GSの通常の正方画素を仮定して `fy = fx` とします。

## 2. 最初は1視点だけRF特徴を抽出する

以下の変数は例です。実在する成功モデルのパスを設定します。

```bash
NEWRF_ROOT=/path/to/NeWRF-ubuntu35
NEWRF_CONFIG=/path/to/successful_run/config.yaml
NEWRF_CKPT=/path/to/successful_run/ckpt.pt
ALIGN_DATA=/workspace/Temp/newrf_alignment

python -m rrf_alignment.extract \
  --newrf-root "$NEWRF_ROOT" \
  --config "$NEWRF_CONFIG" \
  --checkpoint "$NEWRF_CKPT" \
  --views "$ALIGN_DATA/views.json" \
  --model coarse --width 48 --height 32 --carrier-ghz 2.4 \
  --limit 1 --output "$ALIGN_DATA/rf_one_view.pt"
```

最初はcoarseを使います。fineを試す場合は `--model fine` に変更します。
fineはcoarseの密度重みで階層サンプリングし、その位置でfineの中間特徴を取得します。
近端・遠端・サンプル数・inverse-depthは指定YAMLの `sampling` から読みます。
明示的に変える場合は `--near`, `--far`, `--samples`, `--fine-samples`, `--inverse-depth` を指定します。
推論時はサンプリングの摂動を無効にします。

確認する出力:

- `rf_one_view.diagnostics.json`: 生の密度値、正値率、重み総和、特徴ノルム。
- `rf_one_view_preview/mass.png`: 重み総和。表示範囲は常に0～1。
- `rf_one_view_preview/feature_norm.png`: 集約特徴のノルム。表示範囲は同じ場所のJSONに記録。

`mass_mean` と `pooled_feature_norm_mean` がゼロなら、ここで原因を確認します。
学習失敗だけでなく、座標の不一致や、細いRF寄与領域を視野レイが外している可能性があります。
非ゼロでも場の正しさを保証するものではありません。値だけでなく位置・姿勢・画角の対応を確認してください。

## 3. 全視点の特徴と教師 μ を作る

1視点で確認できたら、同じ設定で `--limit` を外します。
各キャッシュは既存ファイルへの上書きを拒否します。

```bash
python -m rrf_alignment.extract \
  --newrf-root "$NEWRF_ROOT" --config "$NEWRF_CONFIG" --checkpoint "$NEWRF_CKPT" \
  --views "$ALIGN_DATA/views.json" --model coarse --width 48 --height 32 \
  --carrier-ghz 2.4 --output "$ALIGN_DATA/rf_features.pt"

VAE_CKPT=/path/to/gray_vae/best.pt
python -m rrf_alignment.encode \
  --views "$ALIGN_DATA/views.json" --vae-checkpoint "$VAE_CKPT" \
  --output "$ALIGN_DATA/visual_mu.pt"
```

VAEチェックポイントに完全な `config` がない場合は、学習時の値を記した
`--vae-config /path/to/vae_config.json` を追加します。形式は `configs/vae.example.json` にあります。
モデルの形状は厳密に照合します。800×1200・潜在32次元という例を無条件に当てはめません。
教師画像は既存VAEと同じグレースケール化・Lanczosリサイズ・0～1正規化を使います。
カメラの画像サイズ、RF特徴マップ、VAE入力の縦横比は一致させます。

## 4. CNNを学習し、復元を見る

```bash
python -m rrf_alignment.train \
  --features "$ALIGN_DATA/rf_features.pt" --targets "$ALIGN_DATA/visual_mu.pt" \
  --output-dir "$ALIGN_DATA/run001" --epochs 200 --batch-size 16 --seed 0

python -m rrf_alignment.decode \
  --predictions "$ALIGN_DATA/run001/predictions.pt" --vae-checkpoint "$VAE_CKPT" \
  --split val --output-dir "$ALIGN_DATA/run001/val_images"
```

設定JSONが必要なVAEでは、decodeにも同じ `--vae-config` を渡します。
比較画像は左から「GS教師画像のグレースケール」「教師 μ のVAE復元」「RF予測 μ のVAE復元」です。
PSNRは表示用の数枚だけでなく、指定splitの全視点で計算します。

- `best.pt`: 検証μ-MSEで選んだCNN、正規化統計、入力モデル・設定の識別情報。
- `history.json`, `metrics.json`: 潜在MSEと、訓練データの平均μを予測するベースライン。
- `splits.json`: 実際の分割。既定は受信機グループ単位でtrain/val/test ≈ 70/20/10%。
- `predictions.pt`: 各視点の予測μと教師μ。

同じ受信機位置の別視点は同じsplitに置きます。別IDでも座標差が1e-5以下なら同一グループとして扱います。
正規化統計はtrainだけで計算します。十分な数の受信機位置が必要です。
空間領域を分けた評価は、manifestの `split` で指定してください。

## 新しい視点で推論する

同じNeWRF・特徴設定で新しいmanifestからRF特徴を抽出し、次を実行します。
この経路では可視画像を読みません。manifestの `image` は対応情報として残して構いません。

```bash
python -m rrf_alignment.predict \
  --checkpoint "$ALIGN_DATA/run001/best.pt" --features "$ALIGN_DATA/new_rf_features.pt" \
  --output "$ALIGN_DATA/predicted_mu.pt"
python -m rrf_alignment.decode \
  --predictions "$ALIGN_DATA/predicted_mu.pt" --vae-checkpoint "$VAE_CKPT" \
  --output-dir "$ALIGN_DATA/new_images"
```

## 解釈と制約

`sum(w)` はNeWRFの密度由来の重みであり、受信電力ではありません。
RFの密度分布は可視物体表面そのものを表すとは限りません。
この特徴はカメラ視野に沿う場のサンプルを集約しますが、深さ方向の情報は集約によって失われます。
場を使う利点の検証には、後で位置・姿勢だけのモデルや未学習NeWRFとの比較が必要です。

今回の目的は、既に学習できたRF場から潜在整合が動くか確かめることです。
CNNのtest位置がNeWRF学習やVAE学習に使われていれば、この分割だけで
「RFも画像も未観測の地点への汎化」を示すことはできません。
後の評価では、NeWRF・VAE・alignmentの学習に使った位置を別々に管理してください。

ソースとチェックポイントのハッシュを保存して取り違えを検出しますが、
そのソースが本当に学習時のものかはチェックポイント単体から検証できません。
成功時の版のNeWRFを指定してください。ReLU/Softplusや合成器の境界処理は変更しません。
現段階では実験用データを接続する前の試作であり、実環境のRF→画像性能は未検証です。

## コードの動作確認

```bash
python -m unittest discover -s tests -v
NEWRF_TEST_ROOT=/path/to/NeWRF python -m unittest discover -s tests -v
```

後者は実際の外部NeWRFコードと合成した重みを使い、coarse/fineの読み込み、
特徴フック、レイ分割の整合性も確認します。実測データの学習精度を評価するテストではありません。
