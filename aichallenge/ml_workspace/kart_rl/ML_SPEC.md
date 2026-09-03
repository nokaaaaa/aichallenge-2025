# kart_rl Machine Learning Spec

この文書は `aichallenge/ml_workspace/kart_rl` の強化学習環境について、入力、出力、報酬関数、終了条件、生成物をまとめたものです。現在のデフォルト設定 `configs/default.yaml` はLiDAR観測版です。

## 目的

固定コース上でレーシングカートを走行させ、以下を満たす方策を学習します。

- 壁、レーン境界に当たらない
- 指定されたゴール区間までなるべく早く走る

学習環境はROS/AWSIMを使わない2D簡易シミュレーションです。Stable-Baselines3のPPOを使い、`uv` で依存関係を管理します。

## 使用データ

### 進捗用コース

`configs/default.yaml` の `track.csv_path` で指定します。

```yaml
track:
  csv_path: "../../workspace/src/aichallenge_submit/simple_trajectory_generator/data/raceline_cctb_30km_wide.csv"
  format: "raceline"
```

このCSVは1周の進捗 `s`、ラップ完了判定、ビューアの参照経路、状態観測版の前方カーブ情報とpure pursuit参照線に使います。

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
  max_speed_mps: 4.165
  max_accel_mps2: 3.2
  max_brake_mps2: 5.0
  max_steer_rad: 0.75
```

制御周期は `env.dt: 0.05` 秒です。

## デフォルト観測: LiDAR

`configs/default.yaml` は、車両先頭LiDARの距離配列を時系列に積み、車速と現在舵角を加えた値を方策入力にします。

```yaml
env:
  type: "lidar"
```

横偏差、方位偏差、参照線曲率は方策の観測には含めません。ただし報酬計算、ラップ判定、壁接触判定には環境内部で参照経路と境界情報を使います。

LiDARは車両先頭中央に取り付けた想定です。

```yaml
lidar:
  angle_min: -1.5666074752807617
  angle_max: 1.5707963705062866
  angle_increment: 0.004188789986073971
  time_increment: 0.00006666666740784422
  scan_time: 0.05000000074505806
  range_min: 0.0
  range_max: 25.0
```

入力側のLiDARビーム数は、デフォルトでは `lidar.sample_ratio: 0.5` により375本です。`lidar.frame_stack: 4` により直近4フレームを古い順に連結し、最後に `speed / max_speed` と正規化舵角を加えます。観測空間は `Box(0, 1, shape=(1502,), dtype=float32)` です。交点がない場合は `range_max` になります。`lidar.sample_ratio` や `lidar.frame_stack` を変えると入力次元も変わります。

## 状態観測版

`configs/state.yaml` を指定すると、LiDARではなく状態量と参照経路情報を観測に使う旧環境で動かせます。

```bash
uv run kart-rl-train --config configs/state.yaml
```

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

デフォルトのLiDAR版では、行動はステア角だけです。速度は固定目標速度として `max_speed_mps` に向かって加速します。

| Index | 名前 | 内容 |
|---:|---|---|
| 0 | steer_ratio | ステア角 |

`action[0]` は `[-1, 1]` から `[-max_steer_rad, max_steer_rad]` に変換します。状態観測版のようなpure pursuit補助は使いません。

`action_dim: 2` にした古い設定では、`action[0]` を `[-1, 1]` から `[min_speed_mps, max_speed_mps]` に変換し、`action[1]` をステア角に使います。

デフォルトLiDAR版の出力先:

```text
models/ppo_kart_lidar.zip
rollouts/latest_lidar.npz
runs/tensorboard_lidar/
```

実行例:

```bash
uv run kart-rl-train
uv run kart-rl-eval
uv run kart-rl-viewer
```

### 状態観測版の行動

状態観測版の行動空間もデフォルトは `Box(-1, 1, shape=(1,), dtype=float32)` です。

| Index | 名前 | 内容 |
|---:|---|---|
| 0 | steer_correction_ratio | pure pursuitステアへの補正 |

### LiDAR版の目標速度

デフォルトの1次元行動では、目標速度は `max_speed_mps` です。2次元行動を使う場合だけ action から決めます。

```text
target_speed = min_speed + 0.5 * (action[0] + 1.0) * (max_speed - min_speed)
```

現在速度から目標速度へ、加速度上限またはブレーキ上限の範囲内で近づけます。

### LiDAR版のステア

LiDAR版のステアはRL出力をそのまま目標ステア角に変換します。

```text
steer_target = action[-1] * max_steer
```

実際のステア角は `max_steer_rate_radps` の範囲内で `steer_target` に近づけます。

### 状態観測版のステア

状態観測版のステアはpure pursuitをベースにし、RLは微小補正だけを学びます。

```text
steer = pure_pursuit_steer + action[-1] * max_steer * max_steer_correction_ratio
```

現在設定では `max_steer_correction_ratio: 0.10` なので、RLが補正できる範囲は最大ステア角の10%です。

既存の保存済みモデルには、古い `Box(-1, 1, shape=(2,), dtype=float32)` の action space で学習したものがあります。その場合も速度側の出力は無視し、最後の action 要素をステアとして使います。

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
  lateral_error: 0.0
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
| ゴール区間到達 | terminated | `finish_on_start_straight_exit: true` の場合、開始直線区間を抜けた |
| 1周完了 | terminated | `finish_on_start_straight_exit: false` の場合、`lap_count >= finish_laps` |
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
  n_steps: 256
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
models/ppo_kart_lidar.zip
```

生成コマンド:

```bash
uv run kart-rl-train
```

### TensorBoardログ

デフォルトLiDAR版:

```text
runs/tensorboard_lidar/
```

状態観測版:

```text
runs/tensorboard/
```

### 評価ロールアウト

デフォルトLiDAR版:

```text
rollouts/latest_lidar.npz
```

状態観測版:

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

LiDAR版の評価ロールアウトには追加で以下を保存します。

| 名前 | 内容 |
|---|---|
| lidar_ranges | 各フレームのLiDAR距離配列。単位はm |
| lidar_angles | 各ビームの車両前方基準角度 |
| lidar_range_max | 最大測距距離 |

古いロールアウトなどで `lidar_ranges` がない場合、ビューアは保存済みの `lane_segments` と各フレームの車両姿勢からLiDARを再計算して表示します。

## ビューア

起動コマンド:

```bash
uv run kart-rl-viewer
```

表示内容:

- 灰色線: `configs/lane.csv` から復元したレーン境界
- 赤線: モデルが実際に走った軌跡
- 青緑の放射線: 車両先頭LiDARの現在スキャン
- 黄色い矩形: 車両

LiDARは入力側で間引いており、間引き率は `lidar.sample_ratio` で指定します。デフォルトは `0.5` です。viewer は評価ロールアウトに保存されたサンプル済みLiDARをそのまま表示します。

操作:

- `space`: 一時停止
- `left/right`: 1秒単位でシーク
- `r`: 最初から再生
- `l`: LiDAR表示のON/OFF

## 現在の評価結果

デフォルトLiDAR版は `uv run kart-rl-train` で学習し、`uv run kart-rl-eval` で `rollouts/latest_lidar.npz` を生成します。

デフォルト設定は固定障害物ありです。`uv run kart-rl-train` は、完走報酬を初期から観測しやすくするため、学習中だけ低速・障害物なしから固定障害物3台へ段階的に上げるカリキュラムを使います。viewer と評価は最終条件の固定障害物3台で実行します。

状態観測版を500kステップ学習した過去の評価例:

```text
time_last 45.75s
laps 1
collision_count 0
stopped_count 0
speed mean 8.15 m/s
```

この状態観測版の結果は `uv run kart-rl-viewer --config configs/state.yaml` で確認できます。
