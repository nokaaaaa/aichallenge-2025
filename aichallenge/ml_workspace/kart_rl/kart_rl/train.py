from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from kart_rl.config import PACKAGE_ROOT, load_config, resolve_path
from kart_rl.env import RacingKartEnv


def make_env(config, seed: int, rank: int):
    def _init():
        env = RacingKartEnv(config)
        env.reset(seed=seed + rank, options={"random_start": True})
        return Monitor(env)

    return _init


def build_model(config, env):
    train_cfg = config["train"]
    algo = train_cfg.get("algorithm", "ppo").lower()
    common = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": float(train_cfg["learning_rate"]),
        "gamma": float(train_cfg["gamma"]),
        "tensorboard_log": str(resolve_path(train_cfg["tensorboard_log"], config)),
        "verbose": 1,
        "seed": int(config.get("seed", 42)),
        "device": train_cfg.get("device", "cuda"),
    }
    if algo == "ppo":
        return PPO(
            **common,
            n_steps=int(train_cfg["n_steps"]),
            batch_size=int(train_cfg["batch_size"]),
            gae_lambda=float(train_cfg["gae_lambda"]),
            clip_range=float(train_cfg["clip_range"]),
        )
    if algo == "sac":
        return SAC(**common, batch_size=int(train_cfg["batch_size"]))
    raise ValueError(f"Unsupported algorithm: {algo}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))

    if args.check_env:
        env = RacingKartEnv(config)
        check_env(env, warn=True)
        print("Gymnasium environment check passed.")
        return

    train_cfg = config["train"]
    n_envs = int(train_cfg.get("n_envs", 1))
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = vec_cls([make_env(config, seed, i) for i in range(n_envs)])
    model = build_model(config, env)
    timesteps = int(args.timesteps or train_cfg["total_timesteps"])
    model.learn(total_timesteps=timesteps, progress_bar=True)

    model_path = resolve_path(train_cfg["model_path"], config)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    env.close()
    print(f"Saved model: {model_path}.zip")


if __name__ == "__main__":
    main()
