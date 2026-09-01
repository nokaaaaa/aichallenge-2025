from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from kart_rl.track import Track, wrap_angle


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float
    steer: float


class RacingKartEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: dict[str, Any], render_mode: str | None = None):
        super().__init__()
        from kart_rl.config import resolve_path

        self.config = config
        track_cfg = config["track"]
        vehicle_cfg = config["vehicle"]
        env_cfg = config["env"]
        track_path = resolve_path(track_cfg["csv_path"], config, must_exist=True)
        track_format = track_cfg.get("format", "raceline")
        if track_format == "lane_boundaries":
            self.track = Track.from_lane_csv(
                track_path,
                half_width_m=track_cfg["half_width_m"],
                curvature_scale=track_cfg.get("curvature_scale", 12.0),
            )
        elif track_format == "raceline":
            self.track = Track.from_csv(
                track_path,
                half_width_m=float(track_cfg["half_width_m"]),
                curvature_scale=track_cfg.get("curvature_scale", 12.0),
            )
        else:
            raise ValueError(f"Unsupported track format: {track_format}")
        self.render_mode = render_mode
        self.dt = float(env_cfg["dt"])
        self.max_episode_steps = int(env_cfg["max_episode_steps"])
        self.finish_laps = int(env_cfg.get("finish_laps", 1))
        self.start_noise_m = float(env_cfg.get("start_noise_m", 0.0))
        self.start_noise_yaw = float(env_cfg.get("start_noise_yaw_rad", 0.0))
        self.min_moving_speed = float(env_cfg.get("min_moving_speed_mps", 0.5))
        self.max_stopped_steps = int(env_cfg.get("max_stopped_steps", 80))
        self.wheelbase = float(vehicle_cfg["wheelbase_m"])
        self.vehicle_width = float(vehicle_cfg["width_m"])
        self.vehicle_length = float(vehicle_cfg["length_m"])
        self.max_speed = float(vehicle_cfg["max_speed_mps"])
        self.min_speed = float(vehicle_cfg["min_speed_mps"])
        self.max_accel = float(vehicle_cfg["max_accel_mps2"])
        self.max_brake = float(vehicle_cfg["max_brake_mps2"])
        self.max_steer = float(vehicle_cfg["max_steer_rad"])
        self.max_steer_rate = float(vehicle_cfg["max_steer_rate_radps"])
        self.reward_cfg = config["reward"]
        self.lookahead = (2.0, 5.0, 10.0, 18.0)
        self.projection_window_m = max(20.0, self.max_speed * self.dt * 10.0)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.ones(8, dtype=np.float32), high=np.ones(8, dtype=np.float32))
        self.state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)
        self.steps = 0
        self.progress_s = 0.0
        self.prev_s = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.lap_count = 0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.stopped_steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        start_s = float(self.np_random.uniform(0.0, self.track.length)) if options and options.get("random_start") else 0.0
        x, y, yaw, _ = self.track.sample_at(start_s)
        lateral = float(self.np_random.normal(0.0, self.start_noise_m))
        x -= lateral * np.sin(yaw)
        y += lateral * np.cos(yaw)
        yaw = float(wrap_angle(yaw + self.np_random.normal(0.0, self.start_noise_yaw)))
        self.state = VehicleState(x=x, y=y, yaw=yaw, speed=1.0, steer=0.0)
        proj = self.track.project(x, y, yaw)
        self.steps = 0
        self.progress_s = 0.0
        self.prev_s = proj.s
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.lap_count = 0
        self.prev_x = self.state.x
        self.prev_y = self.state.y
        self.stopped_steps = 0
        return self._obs(proj), self._info(proj, collision=False)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        accel = self.max_accel * max(float(action[0]), 0.0) + self.max_brake * min(float(action[0]), 0.0)
        steer_rate = self.max_steer_rate * float(action[1])
        self.state.speed = float(np.clip(self.state.speed + accel * self.dt, self.min_speed, self.max_speed))
        self.state.steer = float(np.clip(self.state.steer + steer_rate * self.dt, -self.max_steer, self.max_steer))
        self.state.x += self.state.speed * np.cos(self.state.yaw) * self.dt
        self.state.y += self.state.speed * np.sin(self.state.yaw) * self.dt
        self.state.yaw = float(wrap_angle(self.state.yaw + self.state.speed / self.wheelbase * np.tan(self.state.steer) * self.dt))

        proj = self.track.project_near(
            self.state.x,
            self.state.y,
            self.state.yaw,
            near_s=self.prev_s,
            window_m=self.projection_window_m,
        )
        raw_delta = proj.s - self.prev_s
        if raw_delta < -0.5 * self.track.length:
            raw_delta += self.track.length
        elif raw_delta > 0.5 * self.track.length:
            raw_delta -= self.track.length
        max_step_progress = max(self.state.speed * self.dt * 1.2, 1e-3)
        progress = float(np.clip(raw_delta, -max_step_progress, max_step_progress))
        self.progress_s += max(progress, 0.0)
        self.prev_s = proj.s
        self.steps += 1
        self.lap_count = max(0, int(self.progress_s / self.track.length))
        self.stopped_steps = self.stopped_steps + 1 if self.state.speed < self.min_moving_speed else 0

        margin = 0.5 * self.vehicle_width
        collision = proj.lateral_error < proj.lateral_min + margin or proj.lateral_error > proj.lateral_max - margin
        if collision:
            clamped_lateral = float(np.clip(proj.lateral_error, proj.lateral_min + margin, proj.lateral_max - margin))
            normal = np.array([-np.sin(proj.yaw), np.cos(proj.yaw)])
            self.state.x = float(proj.x + normal[0] * clamped_lateral)
            self.state.y = float(proj.y + normal[1] * clamped_lateral)
            proj = self.track.project_near(
                self.state.x,
                self.state.y,
                self.state.yaw,
                near_s=proj.s,
                window_m=self.projection_window_m,
            )
        lap_finished = self.lap_count >= self.finish_laps
        stopped = self.stopped_steps >= self.max_stopped_steps
        terminated = bool(collision or lap_finished or stopped)
        truncated = bool(self.steps >= self.max_episode_steps)
        distance_moved = float(np.hypot(self.state.x - self.prev_x, self.state.y - self.prev_y))
        reward = self._reward(proj, progress, action, collision, lap_finished, stopped, distance_moved)
        self.prev_action = action.copy()
        self.prev_x = self.state.x
        self.prev_y = self.state.y
        return self._obs(proj), reward, terminated, truncated, self._info(proj, collision=collision)

    def _obs(self, proj) -> np.ndarray:
        obs = np.concatenate(
            [
                np.array(
                    [
                        np.clip(proj.lateral_error / max(abs(proj.lateral_min), abs(proj.lateral_max), 1e-3), -1.0, 1.0),
                        proj.heading_error / np.pi,
                        self.state.speed / self.max_speed,
                        self.state.steer / self.max_steer,
                    ],
                    dtype=np.float32,
                ),
                np.clip(self.track.lookahead_curvatures(proj.s, self.lookahead), -1.0, 1.0),
            ]
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _reward(
        self,
        proj,
        progress: float,
        action: np.ndarray,
        collision: bool,
        lap_finished: bool,
        stopped: bool,
        distance_moved: float,
    ) -> float:
        r = self.reward_cfg
        reward = r["progress"] * progress
        reward += r["speed"] * self.state.speed
        if distance_moved > 1e-6:
            reward -= r.get("wasted_motion", 0.0) * max(distance_moved - max(progress, 0.0), 0.0)
        local_half_width = max(abs(proj.lateral_min), abs(proj.lateral_max), 1e-3)
        reward -= r["lateral_error"] * abs(proj.lateral_error / local_half_width)
        reward -= r["heading_error"] * abs(proj.heading_error / np.pi)
        reward -= r["steer"] * abs(self.state.steer / self.max_steer)
        reward -= r["action_smooth"] * float(np.linalg.norm(action - self.prev_action))
        if self.state.speed < self.min_moving_speed:
            reward -= r.get("low_speed", 0.0)
        if collision:
            reward -= r["wall_collision"]
        if stopped:
            reward -= r.get("stopped", 0.0)
        if lap_finished:
            reward += r["lap_complete"]
        return float(reward)

    def _info(self, proj, collision: bool) -> dict[str, Any]:
        return {
            "x": self.state.x,
            "y": self.state.y,
            "yaw": self.state.yaw,
            "speed": self.state.speed,
            "steer": self.state.steer,
            "s": self.progress_s,
            "lap_progress": (self.progress_s % self.track.length) / self.track.length,
            "lap_count": self.lap_count,
            "lateral_error": proj.lateral_error,
            "lateral_min": proj.lateral_min,
            "lateral_max": proj.lateral_max,
            "heading_error": proj.heading_error,
            "collision": collision,
            "stopped": self.stopped_steps >= self.max_stopped_steps,
            "time": self.steps * self.dt,
        }

    def render(self):
        return None
