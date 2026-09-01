# kart_rl

Python + Gymnasium + Stable-Baselines3 + uv で動く、レーシングカート用のローカル強化学習ワークスペースです。

既存racelineを1周の進捗軸、`configs/lane.csv` を固定レーン情報として使い、既存MPC設定に合わせた簡易キネマティック自転車モデルで学習します。ROS/AWSIMは使わないため、学習ループの動作確認を軽く回せます。

## 構成

```text
kart_rl/
├─ pyproject.toml
├─ configs/default.yaml
├─ configs/lane.csv
├─ kart_rl/
│  ├─ env.py       # Gymnasium環境
│  ├─ track.py     # 固定コース読み込み・中心線射影
│  ├─ train.py     # SB3学習
│  ├─ evaluate.py  # 学習済みモデルで走行ログ生成
│  └─ viewer.py    # Pygameビューア
├─ models/         # 学習済みモデル出力
├─ rollouts/       # ビューア用走行ログ
└─ runs/           # TensorBoardログ
```

## セットアップ

```bash
cd aichallenge/ml_workspace/kart_rl
uv sync
```

環境だけ確認します。

```bash
uv run kart-rl-train --check-env
```

## 学習

```bash
uv run kart-rl-train
```

短く試す場合:

```bash
uv run kart-rl-train --timesteps 5000
```

デフォルトでは `models/ppo_kart.zip` に保存されます。学習デバイスは `configs/default.yaml` の `train.device: "cuda"` でGPUを指定しています。報酬は「中心線方向の進捗」「速度」「1周完了」を正に、「壁接触」「横偏差」「方位偏差」「急な操作」「進捗にならない移動」を負にしています。

## 評価とビューア

学習済みモデルからビューア用ログを生成します。

```bash
uv run kart-rl-eval
```

ビューアを起動します。

```bash
uv run kart-rl-viewer
```

操作:

- `space`: 一時停止
- `left/right`: 1秒単位でシーク
- `r`: 最初から再生

赤線はモデルが実際に走った軌跡です。灰色線は `configs/lane.csv` から復元したレーン境界線です。

## 固定設定

- 進捗軸: `aichallenge/workspace/src/aichallenge_submit/simple_trajectory_generator/data/raceline_cctb_30km_wide.csv`
- レーン表示: `aichallenge/ml_workspace/kart_rl/configs/lane.csv`
- 車両ホイールベース: `1.087 m`
- 車幅: `1.45 m`
- 最大速度: `8.33 m/s`
- 最大加速度: `3.2 m/s^2`
- 最大ブレーキ: `5.0 m/s^2`

設定を変える場合は `configs/default.yaml` を編集します。
