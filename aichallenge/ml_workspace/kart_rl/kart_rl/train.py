from __future__ import annotations

import argparse
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from kart_rl.config import PACKAGE_ROOT, load_config, load_config_for_model, resolve_latest_timestamped_artifact, resolve_path
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


def load_model(config, env, model_path: Path):
    algo = config["train"].get("algorithm", "ppo").lower()
    cls = PPO if algo == "ppo" else SAC
    return cls.load(str(model_path), env=env, device=config["train"].get("device", "cuda"))


def resolve_model_path(model_setting: str | Path, config: dict) -> Path:
    if Path(model_setting).suffix:
        return resolve_path(model_setting, config, must_exist=True)
    return resolve_latest_timestamped_artifact(model_setting, config, ".zip")


class FinishRateCallback(BaseCallback):
    def __init__(self, window_size: int = 100):
        super().__init__()
        self.finished_history: deque[bool] = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                self.finished_history.append(bool(info.get("finished", False)))
        if self.finished_history:
            self.logger.record("rollout/finish_rate_100", float(np.mean(self.finished_history)))
            self.logger.record("rollout/finish_count_100", int(sum(self.finished_history)))
        return True


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


def save_training_artifacts(model: BaseAlgorithm, config: dict, train_cfg: dict) -> Path:
    base_model_path = resolve_path(train_cfg["model_path"], config)
    model_path = make_run_model_path(base_model_path)
    model.save(str(model_path))
    config_out = model_path.parent / "config.yaml"
    with config_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump({key: value for key, value in config.items() if key != "_config_path"}, f, sort_keys=False)
    if train_cfg.get("algorithm", "ppo").lower() == "ppo":
        policy_path = export_ppo_policy_npz(model, Path(model_path))
        update_latest_symlink(policy_path, base_model_path.with_name(f"{base_model_path.name}_latest_policy.npz"))
        print(f"Saved policy: {policy_path}")
    update_latest_symlink(model_path.with_suffix(".zip"), base_model_path.with_name(f"{base_model_path.name}_latest.zip"))
    print(f"Saved config: {config_out}")
    print(f"Saved model: {model_path}.zip")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--resume-model", "--model", dest="resume_model", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    resume_model_path = resolve_model_path(args.resume_model, config) if args.resume_model else None
    if resume_model_path is not None:
        config = load_config_for_model(resume_model_path, config)
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
    if resume_model_path is None:
        model = build_model(config, env)
    else:
        model = load_model(config, env, resume_model_path)
        print(f"Resuming model: {resume_model_path}")
    timesteps = int(args.timesteps or train_cfg["total_timesteps"])
    try:
        try:
            model.learn(
                total_timesteps=timesteps,
                progress_bar=True,
                reset_num_timesteps=resume_model_path is None,
                callback=FinishRateCallback(),
            )
        except KeyboardInterrupt:
            print("Training interrupted. Saving current model before exit...")
        save_training_artifacts(model, config, train_cfg)
    finally:
        try:
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
