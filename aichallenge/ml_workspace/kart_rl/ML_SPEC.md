# kart_rl Machine Learning Spec

この文書は `aichallenge/ml_workspace/kart_rl` の強化学習環境について、入力、出力、報酬関数、終了条件、生成物をまとめたものです。

## 目的

固定コース上でレーシングカートを走行させ、以下を満たす方策を学習します。

- 壁、レーン境界に当たらない
- 1周をなるべく早く完走する

学習環境はROS/AWSIMを使わない2D簡易シミュレーションです。Stable-Baselines3のPPOを使い、`uv` で依存関係を管理します。

## 使用データ

### 進捗用コース

`configs/default.yaml` の `track.csv_path` で指定します。

```yaml
track:
  csv_path: "../../workspace/src/aichallenge_submit/simple_trajectory_generator/data/raceline_cctb_30km_wide.csv"
  format: "raceline"
```

このCSVは1周の進捗 `s`、ラップ完了判定、前方カーブ情報、pure pursuitの参照線に使います。

### レーン境界

`configs/default.yaml` の `track.lane_csv_path` で指定します。

```yaml
track:
  lane_csv_path: "lane.csv"
```

`configs/lane.csv` はビューア表示用のレーン境界として読み込みます。ビューアでは灰色線として表示されます。

## 車両モデル

簡易キネマティック自転車モデルです。

```text
x_dot   = v * cos(yaw)
y_dot   = v * sin(yaw)
yaw_dot = v / wheelbase * tan(steer)
```

主な設定値:

```yaml
vehicle:
  wheelbase_m: 1.087
  width_m: 1.45
  length_m: 1.8
  max_speed_mps: 8.33
  max_accel_mps2: 3.2
  max_brake_mps2: 5.0
  max_steer_rad: 0.75
```

制御周期は `env.dt: 0.05` 秒です。

## 観測

Gymnasiumの観測空間は `Box(-1, 1, shape=(8,), dtype=float32)` です。

| Index | 名前 | 内容 | 正規化 |
|---:|---|---|---|
| 0 | lateral_error | 参照線からの横偏差 | 局所レーン幅相当で割り、`[-1, 1]` にclip |
| 1 | heading_error | 参照線方向と車両yawの差 | `pi` で割る |
| 2 | speed | 車速 | `max_speed_mps` で割る |
| 3 | steer | 現在ステア角 | `max_steer_rad` で割る |
| 4 | curvature_2m | 2m前方の曲率 | `curvature_scale` を掛けてclip |
| 5 | curvature_5m | 5m前方の曲率 | 同上 |
| 6 | curvature_10m | 10m前方の曲率 | 同上 |
| 7 | curvature_18m | 18m前方の曲率 | 同上 |

前方カーブ情報は `track.lookahead_curvatures()` で取得します。

## 行動

Gymnasiumの行動空間は `Box(-1, 1, shape=(2,), dtype=float32)` です。

| Index | 名前 | 内容 |
|---:|---|---|
| 0 | target_speed_ratio | 目標速度 |
| 1 | steer_correction_ratio | pure pursuitステアへの補正 |

### 目標速度

`action[0]` は `[-1, 1]` から `[min_speed_mps, max_speed_mps]` に線形変換されます。

```text
target_speed = min_speed + 0.5 * (action[0] + 1) * (max_speed - min_speed)
```

現在速度から目標速度へ、加速度上限またはブレーキ上限の範囲内で近づけます。

### ステア

ステアはpure pursuitをベースにし、RLは微小補正だけを学びます。

```text
steer = pure_pursuit_steer + action[1] * max_steer * max_steer_correction_ratio
```

現在設定では `max_steer_correction_ratio: 0.10` なので、RLが補正できる範囲は最大ステア角の10%です。これにより、低レベル操舵をゼロから探索するよりも「壁に当たらず速く走る速度選択」に学習を寄せています。

## 報酬関数

1ステップごとの報酬は `RacingKartEnv._reward()` で計算します。

```text
reward =
    progress_weight * progress
  + speed_weight * speed
  - wasted_motion_weight * max(distance_moved - max(progress, 0), 0)
  - lateral_error_weight * abs(lateral_error / local_half_width)
  - heading_error_weight * abs(heading_error / pi)
  - steer_weight * abs(steer / max_steer)
  - action_smooth_weight * norm(action - prev_action)
  - low_speed_penalty       if speed < min_moving_speed
  - wall_collision_penalty  if collision
  - stopped_penalty         if stopped
  + lap_complete_bonus      if lap_finished
```

現在の重み:

```yaml
reward:
  progress: 35.0
  speed: 0.04
  lateral_error: 1.2
  heading_error: 0.5
  steer: 0.01
  action_smooth: 0.02
  wasted_motion: 2.5
  low_speed: 0.08
  stopped: 120.0
  wall_collision: 300.0
  lap_complete: 1000.0
```

### 報酬の意図

`progress` は参照racelineに沿って前進した距離です。1周を速くする主成分です。

`speed` は速度を上げるための小さな加点です。進捗なしで速度だけ上げる挙動を避けるため、主報酬は `progress` に置いています。

`wasted_motion` は実移動距離のうち進捗に変換されていない分へのペナルティです。その場旋回や横滑り的な無駄移動を抑えます。

`lateral_error` と `heading_error` はコース追従性を保つためのペナルティです。

`wall_collision` は壁接触を強く避けるための大きなペナルティです。

`lap_complete` は1周完了時のボーナスです。

## 終了条件

エピソードは以下のいずれかで終了します。

| 条件 | 種別 | 内容 |
|---|---|---|
| 壁接触 | terminated | 横偏差が許容境界を超えた |
| 1周完了 | terminated | `lap_count >= finish_laps` |
| 停止継続 | terminated | `speed < min_moving_speed_mps` が `max_stopped_steps` 続いた |
| 最大ステップ | truncated | `steps >= max_episode_steps` |

現在の設定:

```yaml
env:
  max_episode_steps: 4000
  finish_laps: 1
  min_moving_speed_mps: 0.5
  max_stopped_steps: 120
  boundary_margin_m: 0.15
```

## 学習設定

Stable-Baselines3のPPOを使います。

```yaml
train:
  algorithm: ppo
  device: "cuda"
  total_timesteps: 500000
  n_envs: 4
  learning_rate: 0.0003
  n_steps: 1024
  batch_size: 256
  gamma: 0.995
  gae_lambda: 0.95
  clip_range: 0.2
```

GPUは `device: "cuda"` で指定しています。ただし、現在の方策はMLPなので、Stable-Baselines3は「PPO + MLPではGPU効率が高くない」という警告を出します。

## 出力

### モデル

学習済みモデル:

```text
models/ppo_kart.zip
```

生成コマンド:

```bash
uv run kart-rl-train
```

### TensorBoardログ

```text
runs/tensorboard/
```

### 評価ロールアウト

```text
rollouts/latest.npz
```

生成コマンド:

```bash
uv run kart-rl-eval
```

`frames` 配列には各ステップの情報が入ります。

| Index | 名前 | 内容 |
|---:|---|---|
| 0 | time | シミュレーション時刻 |
| 1 | x | 車両位置x |
| 2 | y | 車両位置y |
| 3 | yaw | 車両yaw |
| 4 | speed | 車速 |
| 5 | steer | ステア角 |
| 6 | s | reset後の累積進捗距離 |
| 7 | lap_progress | 1周内の進捗率 |
| 8 | lateral_error | 横偏差 |
| 9 | collision | 壁接触フラグ |
| 10 | lateral_min | 局所境界の最小横位置 |
| 11 | lateral_max | 局所境界の最大横位置 |
| 12 | stopped | 停止終了フラグ |

## ビューア

起動コマンド:

```bash
uv run kart-rl-viewer
```

表示内容:

- 灰色線: `configs/lane.csv` から復元したレーン境界
- 赤線: モデルが実際に走った軌跡
- 黄色い矩形: 車両

操作:

- `space`: 一時停止
- `left/right`: 1秒単位でシーク
- `r`: 最初から再生

## 現在の評価結果

直近の500kステップ学習後の評価例:

```text
time_last 45.75s
laps 1
collision_count 0
stopped_count 0
speed mean 8.15 m/s
```

この結果は `rollouts/latest.npz` を `kart-rl-viewer` で確認できます。
