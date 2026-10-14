# 特徴マップ型VAEを使う

以前の `views.newrf.json`、受信位置単位の分割、グレースケール化、Lanczosによるサイズ調整、0〜1の画素値をそのまま使う追加実装です。

既存の `vae.py`、`train_vae.py`、ベクトル型VAEの重みを変更しません。新しいモデルの学習結果には `gray_map_vae` という別の種類を記録します。

## 1. sptialVAEへの移設

配布した `move_vae_to_sptialVAE.py` をDocker内の `/workspace` に置き、実行します。

```bash
cd /workspace
python3 move_vae_to_sptialVAE.py --target /workspace
```

前回の追加ファイルが変更されていないことと、移設先に別内容のファイルがないことを確認します。新配置の全ファイルの保存と照合を終えてから、前回の9ファイルを元の配置から取り除きます。どちらかに編集があれば、削除せず停止します。

モデル、学習、評価、潜在の書き出し、設定、説明書、テストをすべて `scripts/newrf_latent_alignment/sptialVAE` 以下へまとめます。既存の補助コードは親フォルダから読み込みます。モデル構造と学習条件は変更しません。

前の `apply_vae_map_update.py` も既知の未編集ファイルなら削除します。学習データや保存した重みには触れません。元のフォルダが他のコードから使われている可能性があるので、フォルダ自体は削除しません。

## 2. 最初の学習

```bash
cd /workspace
ALIGN_DATA=/workspace/temp/alignment_dataset/random2000_4dir_v2000

python3 -u -m scripts.newrf_latent_alignment.sptialVAE.train_vae_map \
  --views "$ALIGN_DATA/views.newrf.json" \
  --vae-config /workspace/scripts/newrf_latent_alignment/sptialVAE/configs/vae.map_f8.json \
  --output-dir "$ALIGN_DATA/vae/map_f8_c4_run001" \
  --epochs 200 \
  --batch-size 2 \
  --lr 1e-4 \
  --recon-loss mse \
  --beta 1e-3 \
  --kl-reduction mean \
  --warmup-epochs 10 \
  --patience 30 \
  --seed 0 \
  --device cuda
```

想定される分割はtrain 1600枚、val 200枚、test 200枚です。実際には既存manifestの指定を使い、分割を作り直しません。同じ受信位置が複数の分割に混じっていたら停止します。学習・モデル選択ではtest画像を開きません。

モデルの形状は以下です。最初の3段はstride 2の畳み込みで、各段に残差ブロックを1個入れます。

```text
入力                     B ×   1 × 800 × 1200
Encoder 1                B ×  32 × 400 ×  600
Encoder 2                B ×  64 × 200 ×  300
Encoder 3                B × 128 × 100 ×  150
1×1 Convの平均 μ         B ×   4 × 100 ×  150
1×1 Convの対数分散       B ×   4 × 100 ×  150
サンプリング後 z         B ×   4 × 100 ×  150
Decoder                  B ×   1 × 800 × 1200
```

デコーダはnearestアップサンプリングと畳み込みを使います。全結合層とFlattenによるボトルネック、EncoderからDecoderへ潜在を迂回する接続はありません。

設定を `vae.map_f16.json` にすると潜在は `4 × 50 × 75` です。設定ファイルの `latent_channels` でチャネル数を変更できます。高さ・幅はダウンサンプリング倍率で割り切れる必要があります。

## 3. 損失とKLの単位

初回は以前と同じ画素平均MSEを使い、次の正規化した目的で学習します。

```text
loss = MSE_pixel_mean + beta(t) × KL_element_mean
KL_sum  = 1画像について潜在のC/H/Wすべてを合計し、バッチで平均
KL_mean = KL_sum / (C × H × W)
```

`beta(t)` は最初の10 epochで0付近から `1e-3` まで線形に増やします。`--warmup-epochs 0` で最初から指定した係数になります。

以前のモデルは `MSE + 1e-4 × KL_sum` でした。今回は潜在が60000要素あるので、その係数を合計KLへそのまま適用すると正則化の尺度が変わります。初回設定の `1e-3 × KL_mean` は `1.6667e-8 × KL_sum` と同じです。これは画質を確認するための初期設定で、最適な係数が確定したわけではありません。

画素と潜在をそれぞれ平均した目的であり、画像尤度と合計KLをそのまま足した標準ELBOの数値とは区別してください。ログには `KL_mean`、`KL_sum`、実際に加算した `weighted_KL` を出します。

`--kl-reduction sum` で以前と同じKL集約方法も選べますが、係数の再調整が必要です。

`--recon-loss l1` も使えます。その場合も以前の結果と比較するため、最良モデルは検証の `decode(mu)` のMSEで選びます。

## 4. 出力と読み方

```text
map_f8_c4_run001/
├── arguments.json / config.json / splits.json
├── best.pt / last.pt
├── history.json
├── best_train_preview.png
├── best_val_preview.png
└── best_val_examples/
    ├── 00_input.png / 00_mu.png / 00_sample.png
    └── 01_input.png / 01_mu.png / 01_sample.png
```

プレビューは左から入力、平均μからの再構成、サンプリングしたzからの再構成です。元の800×1200解像度での細部は `best_val_examples` で確認します。

- `val_mu_mse`：平均μからの再構成MSE。以前の128次元モデルの0.005383と比較する指標です。
- `val_sample_mse`：サンプリングを含めた再構成MSE。評価用乱数は毎回固定します。1回のサンプルによる診断値です。
- `val_mu_psnr_db`：画素範囲0〜1で、検証全体の平均MSEから計算したPSNRです。
- `train_weighted_kl`：実際に学習損失へ加えたKL項の平均です。

最初に、両再構成の差、家具・壁・床の境界、trainとvalの差を確認します。

潜在の総要素数は128から60000へ増えており、残差ブロックとデコーダ方式も変わります。この初回実験だけで「空間構造を保持した効果」だけを分離したとは主張できません。まず十分な再構成が可能かを確認し、その後、潜在要素数や構造を揃えた比較を設計してください。

## 5. AEとの比較

上記の学習コマンドに `--mode ae` を追加し、出力先を `map_f8_c4_ae_run001` のように変えてください。AEモードはサンプリングとKLの両方を無効にします。`--beta 0` だけではVAEモードのサンプリングは残ります。

AEモードの分散ヘッドは学習しません。AEのKL・分散をVAEの分布評価に使わないでください。

## 6. 再開

同じ学習コマンドに `--resume` を追加すると、同じ出力先の `last.pt` から再開します。`--epochs` は追加分でなく通算の上限です。最適化状態、データ順の乱数状態、CPU/CUDAの乱数状態を保存します。モデル・manifest・主要な学習設定が変わった場合は再開を拒否します。

異なるGPU/PyTorch環境にまたがるビット単位の一致は保証しません。CPUでの中断再開が連続実行と一致することをテストしています。

## 7. 評価のみ

```bash
cd /workspace
ALIGN_DATA=/workspace/temp/alignment_dataset/random2000_4dir_v2000

python3 -m scripts.newrf_latent_alignment.sptialVAE.evaluate_vae_map \
  --views "$ALIGN_DATA/views.newrf.json" \
  --vae-checkpoint "$ALIGN_DATA/vae/map_f8_c4_run001/best.pt" \
  --output-dir "$ALIGN_DATA/vae/map_f8_c4_run001/eval_val" \
  --split val \
  --device cuda
```

最終モデルと実験条件を決めた後、`--split test` と新しい出力先で最終評価できます。評価では学習に使ったmanifestと同じであることを確認します。

## 8. 潜在マップを書き出す

```bash
cd /workspace
ALIGN_DATA=/workspace/temp/alignment_dataset/random2000_4dir_v2000

python3 -m scripts.newrf_latent_alignment.sptialVAE.encode_vae_map \
  --views "$ALIGN_DATA/views.newrf.json" \
  --vae-checkpoint "$ALIGN_DATA/vae/map_f8_c4_run001/best.pt" \
  --output-dir "$ALIGN_DATA/vae/map_f8_c4_run001/mean_maps" \
  --split all \
  --device cuda
```

サンプリングしないμを、1視点につき1個の `C × H × W` テンソルとして保存します。`index.json` にview ID、receiver ID、分割、カメラ署名、画像・checkpointのハッシュ、ファイルへの対応を記録します。

この出力は `visual_mean_maps` で、以前のベクトル用 `encode.py` / `train.py` / `RFMapEncoder` には直接入りません。次工程でRF側の出力を空間マップへ対応させる必要があります。マップのチャネルがRFで観測可能な成分と観測困難な成分に自然に分離することも保証されません。

## 検証

標準ライブラリのunittestで、形状と勾配、解析的なKL値、決定論的再構成と確率的サンプリング、AEモード、受信位置をまたぐ分割漏れ、test画像を使わない学習、保存・再開、評価、潜在マップの書き出しを確認します。

```bash
cd /workspace
python3 -m unittest discover -s scripts/newrf_latent_alignment/sptialVAE/tests -p test_spatial_vae.py -v
```

WindowsのRTX 4060 Tiでも、以前の元画像2枚を使った3ステップの800×1200学習が成功しました。ピークのテンソル使用メモリは約2.35 GiBでした。これは動作確認であり、検証画像の再構成品質や汎化性能を示す結果ではありません。

## 参考

- [既存の前処理・カメラ対応コード](https://github.com/AG23015-Inumaru/Radio-radiance-field_latent-alignment/tree/main/scripts/newrf_latent_alignment)
- [AutoencoderKLの参考実装](https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py)
- [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

既存の補助コードは動作確認用に参照しただけで、追加配布の内容には含めていません。
