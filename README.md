# Radio radiance field latent alignment

学習済みNeWRFの特徴を、可視画像のVAE潜在平均 μ に対応付ける実験用リポジトリです。
NeWRFの学習と可視画像用Gaussian Splattingの学習・レンダリングは、それぞれ別リポジトリで行います。
ここでは、その学習済みモデル・生成画像・カメラ設定を入力として使用します。

## ファイルの置き場所

| 場所 | 内容 |
|---|---|
| `scripts/newrf_latent_alignment/` | 今回の特徴抽出・潜在整合・画像復元のPythonコード、依存パッケージ一覧 |
| `scripts/newrf_latent_alignment/tests/` | 上記コードの動作確認 |
| `configs/newrf_latent_alignment/` | カメラ対応表・VAE設定の記入例 |
| `configs/local/` | 自分の環境に合わせた設定。Git管理対象外 |
| `outputs/newrf_latent_alignment/exp001/` | 1実験分の特徴・学習結果・復元画像。Git管理対象外 |
| `docs/newrf_latent_alignment.md` | 実行手順と特徴量の説明 |
| `archive/setup/` | 導入時のパッチと以前の起動スクリプトの控え |
| `Dockerfile/`, `docker_run.sh` | このリポジトリの実行環境 |

Pythonコードは `scripts/` に置きます。生成されるファイルは `outputs/` にまとめ、
実験を変える際は `exp001` を `exp002` などに変えます。
出力先は各コマンドの `--output` / `--output-dir` で指定します。

元データと外部モデルは別リポジトリ側のパスを参照します。
このリポジトリの `dataset/` などに既に置いた手元のデータは、今回の整理では移動・削除しません。
コンテナを使う場合は、外部ファイルをコンテナ内から参照できるようにマウントしてください。

## 最初の確認

以下のコマンドは、このリポジトリのルート（コンテナ内では `/workspace`）から実行します。

```bash
python -m pip install -r scripts/newrf_latent_alignment/requirements.txt
python -m scripts.newrf_latent_alignment.extract --help
```

PyTorchは使用しているCUDA環境に合うものを利用してください。

次は、成功したNeWRFのチェックポイント・学習時YAML・ソースを指定し、1視点分のRF特徴を抽出します。
[詳しい実行手順](docs/newrf_latent_alignment.md)に沿って進めてください。

## 可視画像の入力

このリポジトリには可視画像用Gaussian Splattingのコードを置きません。
別リポジトリで用意した次のファイルを参照します。

- レンダリング済みの可視画像
- **その画像をレンダリングした位置・姿勢・画角**を記録したカメラ設定
- GSのワールド座標からNeWRF座標への変換

`prepare_views.py` は、上記の画像とカメラ設定を潜在整合用の対応表に変換するコードです。
GSを学習・レンダリングするコードではないため、`scripts/newrf_latent_alignment/` に残しています。
この試作は既存のグレースケールVAEを使用します。
