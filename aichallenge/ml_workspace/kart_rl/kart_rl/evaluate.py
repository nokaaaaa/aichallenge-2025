from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from kart_rl.config import PACKAGE_ROOT, load_config, resolve_path
from kart_rl.env import RacingKartEnv


def load_model(config, env, model_path: Path):
    algo = config["train"].get("algorithm", "ppo").lower()
    cls = PPO if algo == "ppo" else SAC
    return cls.load(str(model_path), env=env, device=config["train"].get("device", "cuda"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    env = RacingKartEnv(config)
    default_model = config["viewer"].get("model_path") or (config["train"]["model_path"] + ".zip")
    model_path = resolve_path(args.model or default_model, config, must_exist=True)
    out_path = resolve_path(args.out or config["viewer"]["rollout_path"], config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(config, env, model_path)
    obs, info = env.reset(seed=int(config.get("seed", 42)))
    frames = []
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=args.deterministic)
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
            ]
        )

    np.savez_compressed(
        out_path,
        frames=np.asarray(frames, dtype=np.float32),
        track=env.track.points.astype(np.float32),
        left_boundary=env.track.left_boundary.astype(np.float32)
        if env.track.left_boundary is not None
        else np.empty((0, 2), dtype=np.float32),
        right_boundary=env.track.right_boundary.astype(np.float32)
        if env.track.right_boundary is not None
        else np.empty((0, 2), dtype=np.float32),
        half_width=np.float32(env.track.half_width_m),
        vehicle_length=np.float32(env.vehicle_length),
        vehicle_width=np.float32(env.vehicle_width),
        total_reward=np.float32(total_reward),
    )
    env.close()
    print(f"Saved rollout: {out_path}")
    print(f"reward={total_reward:.2f} time={info['time']:.2f}s laps={info['lap_count']} collision={info['collision']}")


if __name__ == "__main__":
    main()
