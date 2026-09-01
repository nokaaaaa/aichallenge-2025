from __future__ import annotations

from typing import Any

from kart_rl.env import RacingKartEnv
from kart_rl.lidar_env import LidarRacingKartEnv


def make_racing_env(config: dict[str, Any]):
    env_type = config.get("env", {}).get("type", "lidar")
    if env_type == "state":
        return RacingKartEnv(config)
    if env_type == "lidar":
        return LidarRacingKartEnv(config)
    raise ValueError(f"Unsupported env.type: {env_type}")
