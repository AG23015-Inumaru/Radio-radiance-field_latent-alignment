# 深さ区間を残すRF → f32 latent alignment（案1）

NeWRFのレイ上の中間特徴を4つの距離区間に分けて保持し、画素ごとに距離区間を選択する重みを学習して、画像VAEのμ（4×25×37）を予測します。

## 配置と実行

ZIPを展開し、`depth_alignment_f32` フォルダをそのまま次へ置いてください。フォルダ内の2つのPythonファイルを一緒に配置します。

```text
/workspace/scripts/newrf_latent_alignment/alignment/depth_alignment_f32/
    run_alignment_depth_f32.py
    alignment_common_f32.py
    README.md
```

Docker内で実行します。

```bash
cd /workspace
python3 -u scripts/newrf_latent_alignment/alignment/depth_alignment_f32/run_alignment_depth_f32.py \
  --workspace /workspace
```

前回の実行条件を読み込み、深さ別RF特徴の再抽出 → CNN学習 → 検証画像の保存まで進みます。実データの再抽出・学習はこのコマンドから開始します。

## 前回から引き継ぐもの

既定のデータセットは次です。

```text
/workspace/temp/alignment_dataset/random2000_4dir_v2000
```

`runs/align_f32_c4_run001/arguments.json` から、RFキャッシュ・VAE・視点JSON・教師μ・学習条件を読みます。既定は最大200 epoch、バッチ16、AdamW、学習率0.001、weight decay 0.0001、patience 30、seed 0です。既存のtrain/val/testの受信位置単位の分割を使います。

過去の抽出ログから、次のNeWRFパスを既定値に設定しています。

| 用途 | `/workspace/` からのパス |
|---|---|
| NeWRFソース | `dataset/NEWRF/rf3dgs_rx_cos0999_sameinit_100k/source` |
| 設定 | `dataset/NEWRF/rf3dgs_rx_cos0999_sameinit_100k/config/j4ru0_effective.yaml` |
| 重み | `dataset/NEWRF/rf3dgs_rx_cos0999_sameinit_100k/checkpoint/nist_rf3dgs_J4RU0/ckpt.pt` |

このNIST実験の過去ログではcoarseを使用しています。実行時には、前回のRFキャッシュが記録したcoarse/fineの選択を引き継ぎ、モデル・設定・ソースのハッシュを照合します。サンプリング数、near/far、inverse-depth、周波数、RFマップの解像度も前回の値を使います。

別の場所へ移動した場合は `--newrf-root`、`--newrf-config`、`--newrf-checkpoint` で指定できます。前回のalignment保存先が異なる場合は `--baseline-run` を指定します。

## RF特徴とモデル

集約済みの65×32×48から距離情報を復元することはできないため、学習済みNeWRFに再度問い合わせます。NeWRFと画像VAEの重みは固定します。

各距離区間kに、次の値を保持します。

```text
中間特徴：Σ w_i h_i             64チャネル（標準モデルの場合）
重み総和：Σ w_i                 1チャネル
代表距離：Σ w_i z_i / Σ w_i / far   1チャネル
```

1視点の保存形状は、**4区間 × 66チャネル × 高さ32 × 幅48**です。区間数は `--depth-bins` で変更できます。数値の保存精度はfloat32です。

区間境界は `[0, far]` を等間隔に4分割します。NeWRFのサンプラーが最初の点を距離約0へ置く実装にも対応します。重みが0の区間は代表距離を区間中央に置き、重み総和0と組み合わせて区別します。**この距離はRF場のサンプル位置までの距離で、可視物体の表面深度ではありません。**

```text
4つの距離区間のRF特徴
        ↓ 区間共通のCNN＋距離位置の埋め込み
各画素・各区間の特徴
        ↓ 学習する選択重み（区間方向のsoftmax）
2D特徴マップ
        ↓ 空間CNN
予測μ：4×25×37
```

重み総和0の区間は選択対象から除きます。全区間が0のレイでは、有限な重みを使うフォールバックを設けています。

損失は前回と同じく元のVAE座標でのμ-MSEです。入力・出力の正規化統計はtrainだけから求め、最良モデルはvalのμ-MSEで選びます。testは分割に保持します。

## 保存先と比較

再抽出したRF特徴：

```text
/workspace/temp/alignment_dataset/random2000_4dir_v2000/rf/depth4_run001/
```

各視点を別の `.pt` に保存します。再抽出が途中で止まった場合、同じコマンドで保存済みの視点を再利用できます。設定やモデルの識別情報が異なるキャッシュへの追記は拒否します。

学習結果：

```text
/workspace/temp/alignment_dataset/random2000_4dir_v2000/runs/align_depth4_f32_c4_run001/
```

| ファイル | 内容 |
|---|---|
| `best.pt` | 最良モデル、正規化統計、モデル設定、入力の識別情報 |
| `history.json` | 各epochのtrain・val潜在MSE |
| `metrics.json` | 最良モデルの誤差、全val画像のMSE・PSNR、前回との差 |
| `best_val_preview.png` | 元画像・教師μの復元・予測μの復元 |
| `train_with_mean_baseline.png` | train先頭4枚の、元画像・教師μ・train平均μ・予測μの復元 |
| `val_with_mean_baseline.png` | val先頭4枚の同じ比較 |
| `val_rf_bin_attention.png` | val先頭4枚の各RF距離区間の選択重み。RF重みが全て0のレイは黒 |
| `predictions.pt` | train・valの教師μと予測μ |
| `arguments.json`, `splits.json`, `model_config.json` | 実行設定・分割・モデル構成 |

前回の `best.pt` に記録されたVAE・対応表・RFキャッシュの識別情報が一致する場合、`metrics.json` の `reference_alignment` に前回の結果と差を保存します。

前回共有された比較基準は、val μ-MSE **0.277993**、画像PSNR **18.77 dB**です。教師μの画像PSNRは33.19 dBでした。

## 容量と再実行

2000視点、4区間、64次元中間特徴の場合、RF特徴テンソルは約3.24 GBです。既定では学習時に約3.02 GiBのCPUメモリへ読み込みます。追加でモデルや作業領域のメモリを使用します。

メモリを節約する場合は、ファイルを都度読む方式に変更できます。

```bash
python3 -u scripts/newrf_latent_alignment/alignment/depth_alignment_f32/run_alignment_depth_f32.py \
  --workspace /workspace --cache-mode disk
```

NAS上のキャッシュを都度読む場合は、RAM方式より学習が遅くなる可能性があります。

抽出だけ行う場合は `--extract-only`、完成済みキャッシュで学習する場合は `--train-only` を追加します。

既存の学習結果には上書きしません。再学習は `--output-dir` で新しい保存先を指定します。

## 確認済みの範囲

合成チェックポイントと元のNeWRF実装を用い、coarse/fine・通常/inverse-depthのサンプリングで、距離区間の合計が元のRF特徴を再現することを確認しました。μの対応付け、学習・画像復元、固定重みの保持、キャッシュ再利用、ゼロ重み時の有限な出力、RAM/diskアクセスも確認しています。実データでの改善幅はDocker側の実験結果で判断します。
