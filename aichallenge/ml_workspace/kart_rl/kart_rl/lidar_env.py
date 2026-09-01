from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from kart_rl.env import RacingKartEnv


class LidarRacingKartEnv(RacingKartEnv):
    """Racing kart environment whose policy observation is front-mounted LiDAR only."""

    def __init__(self, config: dict[str, Any], render_mode: str | None = None):
        super().__init__(config, render_mode=render_mode)
        lidar_cfg = config["lidar"]
        self.angle_min = float(lidar_cfg["angle_min"])
        self.angle_max = float(lidar_cfg["angle_max"])
        self.angle_increment = float(lidar_cfg["angle_increment"])
        self.time_increment = float(lidar_cfg["time_increment"])
        self.scan_time = float(lidar_cfg["scan_time"])
        self.range_min = float(lidar_cfg["range_min"])
        self.range_max = float(lidar_cfg["range_max"])
        self.ray_chunk_size = int(lidar_cfg.get("ray_chunk_size", 64))
        self.angles = np.arange(self.angle_min, self.angle_max + 0.5 * self.angle_increment, self.angle_increment)
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.angles), dtype=np.float32),
            high=np.ones(len(self.angles), dtype=np.float32),
            dtype=np.float32,
        )
        if self.track.lane_segments is None or len(self.track.lane_segments) == 0:
            raise ValueError("LidarRacingKartEnv requires track.lane_csv_path")
        self.lane_p0 = self.track.lane_segments[:, 0, :]
        self.lane_v = self.track.lane_segments[:, 1, :] - self.track.lane_segments[:, 0, :]

    def _obs(self, proj) -> np.ndarray:
        ranges = self._scan()
        return (ranges / self.range_max).astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        target_speed = self.min_speed + 0.5 * (float(action[0]) + 1.0) * (self.max_speed - self.min_speed)
        speed_error = target_speed - self.state.speed
        accel_limit = self.max_accel if speed_error >= 0.0 else self.max_brake
        speed_step = float(np.clip(speed_error, -accel_limit * self.dt, accel_limit * self.dt))
        self.state.speed = float(np.clip(self.state.speed + speed_step, self.min_speed, self.max_speed))
        self.state.steer = float(np.clip(float(action[1]) * self.max_steer, -self.max_steer, self.max_steer))

    def _scan(self) -> np.ndarray:
        sensor_x = self.state.x + 0.5 * self.vehicle_length * np.cos(self.state.yaw)
        sensor_y = self.state.y + 0.5 * self.vehicle_length * np.sin(self.state.yaw)
        origin = np.array([sensor_x, sensor_y], dtype=np.float64)
        ranges = np.full(len(self.angles), self.range_max, dtype=np.float64)

        midpoint = self.lane_p0 + 0.5 * self.lane_v
        seg_radius = 0.5 * np.linalg.norm(self.lane_v, axis=1)
        nearby = np.linalg.norm(midpoint - origin, axis=1) <= self.range_max + seg_radius
        lane_p0 = self.lane_p0[nearby]
        lane_v = self.lane_v[nearby]
        if len(lane_p0) == 0:
            return ranges.astype(np.float32)

        ray_angles = self.state.yaw + self.angles
        rays = np.column_stack([np.cos(ray_angles), np.sin(ray_angles)])
        rel = lane_p0 - origin
        for start in range(0, len(rays), self.ray_chunk_size):
            ray_chunk = rays[start : start + self.ray_chunk_size]
            hit = _ray_segment_distances_batch(rel, ray_chunk, lane_v)
            ranges[start : start + len(ray_chunk)] = np.minimum(
                ranges[start : start + len(ray_chunk)],
                hit,
            )
        return ranges


def _ray_segment_distances_batch(rel: np.ndarray, rays: np.ndarray, seg_v: np.ndarray) -> np.ndarray:
    denom = _cross(rays[:, None, :], seg_v[None, :, :])
    valid = np.abs(denom) > 1e-9
    t = np.full_like(denom, np.inf, dtype=np.float64)
    u = np.full_like(denom, np.inf, dtype=np.float64)
    np.divide(_cross(rel[None, :, :], seg_v[None, :, :]), denom, out=t, where=valid)
    np.divide(_cross(rel[None, :, :], rays[:, None, :]), denom, out=u, where=valid)
    hit = valid & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    distances = np.where(hit, t, np.inf).min(axis=1)
    return np.clip(distances, 0.0, np.inf)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
