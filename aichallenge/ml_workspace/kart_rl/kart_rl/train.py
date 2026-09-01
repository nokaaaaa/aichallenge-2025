from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from kart_rl.config import PACKAGE_ROOT, load_config, resolve_path
from kart_rl.env_factory import make_racing_env


def make_env(config, seed: int, rank: int):
    def _init():
        env = make_racing_env(config)
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


def export_ppo_policy_npz(model: PPO, model_path: Path) -> Path:
    policy_path = model_path.with_name(f"{model_path.name}_policy.npz")
    state = {
        key: value.detach().cpu().numpy()
        for key, value in model.policy.state_dict().items()
        if key.startswith(("mlp_extractor.policy_net.", "action_net."))
    }
    np.savez(policy_path, **state)
    return policy_path


def make_run_model_path(base_model_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_model_path.parent / f"{base_model_path.name}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_model_path.parent / f"{base_model_path.name}_{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir / base_model_path.name


def update_latest_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))

    if args.check_env:
        env = make_racing_env(config)
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

    base_model_path = resolve_path(train_cfg["model_path"], config)
    model_path = make_run_model_path(base_model_path)
    model.save(str(model_path))
    config_out = model_path.parent / "config.yaml"
    shutil.copy2(config["_config_path"], config_out)
    if train_cfg.get("algorithm", "ppo").lower() == "ppo":
        policy_path = export_ppo_policy_npz(model, Path(model_path))
        update_latest_symlink(policy_path, base_model_path.with_name(f"{base_model_path.name}_latest_policy.npz"))
        print(f"Saved policy: {policy_path}")
    update_latest_symlink(model_path.with_suffix(".zip"), base_model_path.with_name(f"{base_model_path.name}_latest.zip"))
    print(f"Saved config: {config_out}")
    env.close()
    print(f"Saved model: {model_path}.zip")


if __name__ == "__main__":
    main()
