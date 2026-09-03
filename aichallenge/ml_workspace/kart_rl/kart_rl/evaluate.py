from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from kart_rl.config import PACKAGE_ROOT, load_config, load_config_for_model, resolve_latest_timestamped_artifact, resolve_path
from kart_rl.env_factory import make_racing_env


def load_model(config, env, model_path: Path):
    algo = config["train"].get("algorithm", "ppo").lower()
    cls = PPO if algo == "ppo" else SAC
    return cls.load(str(model_path), env=env, device=config["train"].get("device", "cuda"))


def resolve_model_path(model_setting: str | Path, config: dict) -> Path:
    if Path(model_setting).suffix:
        return resolve_path(model_setting, config, must_exist=True)
    return resolve_latest_timestamped_artifact(model_setting, config, ".zip")


def collect_rollout_data(
    config: dict,
    model_path: Path,
    deterministic: bool = True,
    seed: int | None = None,
    random_seed: bool = False,
) -> tuple[dict, float, dict]:
    env = make_racing_env(config)
    try:
        model = load_model(config, env, model_path)
        reset_seed = None if random_seed else int(config.get("seed", 42) if seed is None else seed)
        obs, info = env.reset(seed=reset_seed)
        frames = []
        lidar_scans = []
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frames.append(
                [
                    info["time"],
                    info["x"],
                    info["y"],
                    info["yaw"],
                    info["speed"],
                    info["steer"],
                    info["s"],
                    info["lap_progress"],
                    info["lateral_error"],
                    float(info["collision"]),
                    info["lateral_min"],
                    info["lateral_max"],
                    float(info["stopped"]),
                    float(info["finished"]),
                ]
            )
            if config.get("env", {}).get("type") == "lidar":
                lidar_scans.append(np.asarray(obs, dtype=np.float32))

        save_data = {}
        if lidar_scans:
            lidar_cfg = config["lidar"]
            sample_ratio = float(lidar_cfg.get("sample_ratio", 1.0))
            frame_stack = max(1, int(lidar_cfg.get("frame_stack", 1)))
            angle_min = float(lidar_cfg["angle_min"])
            angle_max = float(lidar_cfg["angle_max"])
            angle_increment = float(lidar_cfg["angle_increment"])
            full_angles = np.arange(angle_min, angle_max + 0.5 * angle_increment, angle_increment, dtype=np.float32)
            sample_stride = max(1, int(round(1.0 / max(sample_ratio, 1e-3))))
            lidar_angles = full_angles[::sample_stride]
            obs_array = np.asarray(lidar_scans, dtype=np.float32)
            lidar_total = len(lidar_angles) * frame_stack
            lidar_scan_array = obs_array[:, :lidar_total][:, -len(lidar_angles) :]
            save_data.update(
                lidar_ranges=lidar_scan_array * float(lidar_cfg["range_max"]),
                lidar_angles=lidar_angles,
                lidar_range_max=np.float32(lidar_cfg["range_max"]),
            )

        data = dict(
            frames=np.asarray(frames, dtype=np.float32),
            model_path=str(model_path),
            model_dir_name=model_path.parent.name,
            track=env.track.points.astype(np.float32),
            left_boundary=env.track.left_boundary.astype(np.float32)
            if env.track.left_boundary is not None
            else np.empty((0, 2), dtype=np.float32),
            right_boundary=env.track.right_boundary.astype(np.float32)
            if env.track.right_boundary is not None
            else np.empty((0, 2), dtype=np.float32),
            lane_segments=env.track.lane_segments.astype(np.float32)
            if env.track.lane_segments is not None
            else np.empty((0, 2, 2), dtype=np.float32),
            half_width=np.float32(env.track.half_width_m),
            vehicle_length=np.float32(env.vehicle_length),
            vehicle_width=np.float32(env.vehicle_width),
            obstacle_vehicles=np.asarray(
                [[vehicle.x, vehicle.y, vehicle.yaw, vehicle.s, vehicle.lateral] for vehicle in env.obstacle_vehicles],
                dtype=np.float32,
            ),
            total_reward=np.float32(total_reward),
            **save_data,
        )
        return data, total_reward, info
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model_setting = args.model or config["viewer"].get("model_path") or config["train"]["model_path"]
    model_path = resolve_model_path(model_setting, config)
    config = load_config_for_model(model_path, config)
    out_path = resolve_path(args.out or config["viewer"]["rollout_path"], config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data, total_reward, info = collect_rollout_data(config, model_path, args.deterministic)
    np.savez_compressed(
        out_path,
        **data,
    )
    print(f"Loaded model: {model_path}")
    print(f"Saved rollout: {out_path}")
    print(f"reward={total_reward:.2f} time={info['time']:.2f}s laps={info['lap_count']} collision={info['collision']}")


if __name__ == "__main__":
    main()
