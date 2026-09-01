from __future__ import annotations

from collections import deque
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
                lane_csv_path=resolve_path(track_cfg["lane_csv_path"], config, must_exist=True)
                if track_cfg.get("lane_csv_path")
                else None,
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
        self.boundary_margin = float(env_cfg.get("boundary_margin_m", 0.15))
        self.localization_delay_sec = max(0.0, float(env_cfg.get("localization_delay_sec", 0.5)))
        self.steering_delay_sec = max(0.0, float(env_cfg.get("steering_delay_sec", 0.2)))
        self.lookahead_base = float(env_cfg.get("pure_pursuit_lookahead_base_m", 2.0))
        self.lookahead_speed_gain = float(env_cfg.get("pure_pursuit_lookahead_speed_gain", 0.8))
        self.max_steer_correction_ratio = float(env_cfg.get("max_steer_correction_ratio", 0.35))
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

        # action = [target_speed_ratio, steer_correction_ratio]
        # target_speed_ratio: -1..1 maps to min_speed..max_speed
        # steer_correction_ratio adjusts the pure-pursuit steering baseline.
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
        self.commanded_steer = 0.0
        self.localized_state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)
        self.localized_prev_s = 0.0
        self._state_history: deque[tuple[float, VehicleState]] = deque()
        self._steer_command_history: deque[tuple[float, float]] = deque()

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
        self.commanded_steer = 0.0
        self.localized_state = self._copy_state(self.state)
        self.localized_prev_s = proj.s
        self._state_history.clear()
        self._state_history.append((0.0, self._copy_state(self.state)))
        self._steer_command_history.clear()
        self._steer_command_history.append((0.0, self.commanded_steer))
        return self._obs(proj), self._info(proj, collision=False, localized_proj=proj)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._apply_action(action)
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

        collision = self._is_collision(proj)
        if collision:
            proj = self._resolve_collision(proj)
        self._append_state_history(self.steps * self.dt, self.state)
        localized_proj = self._update_localized_state(self.steps * self.dt)
        lap_finished = self.lap_count >= self.finish_laps
        stopped = self.stopped_steps >= self.max_stopped_steps
        terminated = bool(collision or lap_finished or stopped)
        truncated = bool(self.steps >= self.max_episode_steps)
        distance_moved = float(np.hypot(self.state.x - self.prev_x, self.state.y - self.prev_y))
        reward = self._reward(proj, progress, action, collision, lap_finished, stopped, distance_moved)
        self.prev_action = action.copy()
        self.prev_x = self.state.x
        self.prev_y = self.state.y
        return self._obs(localized_proj), reward, terminated, truncated, self._info(proj, collision=collision, localized_proj=localized_proj)

    def _is_collision(self, proj) -> bool:
        margin = self.boundary_margin
        return proj.lateral_error < proj.lateral_min + margin or proj.lateral_error > proj.lateral_max - margin

    def _resolve_collision(self, proj):
        margin = self.boundary_margin
        clamped_lateral = float(np.clip(proj.lateral_error, proj.lateral_min + margin, proj.lateral_max - margin))
        normal = np.array([-np.sin(proj.yaw), np.cos(proj.yaw)])
        self.state.x = float(proj.x + normal[0] * clamped_lateral)
        self.state.y = float(proj.y + normal[1] * clamped_lateral)
        return self.track.project_near(
            self.state.x,
            self.state.y,
            self.state.yaw,
            near_s=proj.s,
            window_m=self.projection_window_m,
        )

    def _apply_action(self, action: np.ndarray) -> None:
        target_speed = self.min_speed + 0.5 * (float(action[0]) + 1.0) * (self.max_speed - self.min_speed)
        speed_error = target_speed - self.state.speed
        accel_limit = self.max_accel if speed_error >= 0.0 else self.max_brake
        speed_step = float(np.clip(speed_error, -accel_limit * self.dt, accel_limit * self.dt))
        self.state.speed = float(np.clip(self.state.speed + speed_step, self.min_speed, self.max_speed))
        base_steer = self._pure_pursuit_steer(self.localized_prev_s)
        steer_correction = float(action[1]) * self.max_steer * self.max_steer_correction_ratio
        self._set_commanded_steer(base_steer + steer_correction)

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

    def _pure_pursuit_steer(self, s: float) -> float:
        lookahead = self.lookahead_base + self.lookahead_speed_gain * self.state.speed
        target_x, target_y, _, _ = self.track.sample_at(s + lookahead)
        target_angle = np.arctan2(target_y - self.localized_state.y, target_x - self.localized_state.x)
        alpha = wrap_angle(target_angle - self.localized_state.yaw)
        steer = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), lookahead)
        return float(np.clip(steer, -self.max_steer, self.max_steer))

    def _append_state_history(self, time_sec: float, state: VehicleState) -> None:
        self._state_history.append((time_sec, self._copy_state(state)))
        earliest_needed = time_sec - self.localization_delay_sec - self.dt
        while len(self._state_history) > 2 and self._state_history[1][0] <= earliest_needed:
            self._state_history.popleft()

    def _set_commanded_steer(self, steer: float) -> None:
        time_sec = self.steps * self.dt
        self.commanded_steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        self._append_steer_command(time_sec, self.commanded_steer)
        self.state.steer = self._delayed_steer(time_sec - self.steering_delay_sec)

    def _append_steer_command(self, time_sec: float, steer: float) -> None:
        self._steer_command_history.append((time_sec, float(steer)))
        earliest_needed = time_sec - self.steering_delay_sec - self.dt
        while len(self._steer_command_history) > 2 and self._steer_command_history[1][0] <= earliest_needed:
            self._steer_command_history.popleft()

    def _delayed_steer(self, target_time_sec: float) -> float:
        if not self._steer_command_history:
            return self.commanded_steer
        delayed_steer = self._steer_command_history[0][1]
        for command_time, command_steer in self._steer_command_history:
            if command_time > target_time_sec:
                break
            delayed_steer = command_steer
        return float(delayed_steer)

    def _update_localized_state(self, time_sec: float):
        self.localized_state = self._delayed_state(time_sec - self.localization_delay_sec)
        localized_proj = self.track.project_near(
            self.localized_state.x,
            self.localized_state.y,
            self.localized_state.yaw,
            near_s=self.localized_prev_s,
            window_m=self.projection_window_m,
        )
        self.localized_prev_s = localized_proj.s
        return localized_proj

    def _delayed_state(self, target_time_sec: float) -> VehicleState:
        if not self._state_history:
            return self._copy_state(self.state)
        if target_time_sec <= self._state_history[0][0]:
            return self._copy_state(self._state_history[0][1])
        history_iter = iter(self._state_history)
        prev_time, prev_state = next(history_iter)
        for next_time, next_state in history_iter:
            t0, s0 = prev_time, prev_state
            t1, s1 = next_time, next_state
            if t0 <= target_time_sec <= t1:
                ratio = (target_time_sec - t0) / max(t1 - t0, 1e-9)
                return VehicleState(
                    x=float(s0.x + ratio * (s1.x - s0.x)),
                    y=float(s0.y + ratio * (s1.y - s0.y)),
                    yaw=float(wrap_angle(s0.yaw + ratio * wrap_angle(s1.yaw - s0.yaw))),
                    speed=float(s0.speed + ratio * (s1.speed - s0.speed)),
                    steer=float(s0.steer + ratio * (s1.steer - s0.steer)),
                )
            prev_time, prev_state = next_time, next_state
        return self._copy_state(self._state_history[-1][1])

    @staticmethod
    def _copy_state(state: VehicleState) -> VehicleState:
        return VehicleState(
            x=float(state.x),
            y=float(state.y),
            yaw=float(state.yaw),
            speed=float(state.speed),
            steer=float(state.steer),
        )

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

    def _info(self, proj, collision: bool, localized_proj=None) -> dict[str, Any]:
        if localized_proj is None:
            localized_proj = proj
        return {
            "x": self.state.x,
            "y": self.state.y,
            "yaw": self.state.yaw,
            "localized_x": self.localized_state.x,
            "localized_y": self.localized_state.y,
            "localized_yaw": self.localized_state.yaw,
            "speed": self.state.speed,
            "steer": self.state.steer,
            "commanded_steer": self.commanded_steer,
            "s": self.progress_s,
            "lap_progress": (self.progress_s % self.track.length) / self.track.length,
            "lap_count": self.lap_count,
            "lateral_error": proj.lateral_error,
            "localized_lateral_error": localized_proj.lateral_error,
            "lateral_min": proj.lateral_min,
            "lateral_max": proj.lateral_max,
            "heading_error": proj.heading_error,
            "localized_heading_error": localized_proj.heading_error,
            "collision": collision,
            "stopped": self.stopped_steps >= self.max_stopped_steps,
            "time": self.steps * self.dt,
        }

    def render(self):
        return None
