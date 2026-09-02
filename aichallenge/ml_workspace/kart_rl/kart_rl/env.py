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


@dataclass(frozen=True)
class ObstacleVehicle:
    s: float
    x: float
    y: float
    yaw: float
    lateral: float


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
        self.finish_on_start_straight_exit = bool(env_cfg.get("finish_on_start_straight_exit", False))
        self.start_noise_m = float(env_cfg.get("start_noise_m", 0.0))
        self.start_noise_yaw = float(env_cfg.get("start_noise_yaw_rad", 0.0))
        self.start_pose_awsim = env_cfg.get("start_pose_awsim")
        self.start_s = self._resolve_start_s()
        self.min_moving_speed = float(env_cfg.get("min_moving_speed_mps", 0.5))
        self.max_stopped_steps = int(env_cfg.get("max_stopped_steps", 80))
        self.boundary_margin = float(env_cfg.get("boundary_margin_m", 0.15))
        self.obstacle_vehicle_count = int(env_cfg.get("obstacle_vehicle_count", 10))
        self.obstacle_min_gap_m = float(env_cfg.get("obstacle_min_gap_m", 8.0))
        self.obstacle_start_clearance_m = float(env_cfg.get("obstacle_start_clearance_m", 8.0))
        self.obstacle_lateral_margin_m = float(env_cfg.get("obstacle_lateral_margin_m", 0.25))
        self.obstacle_placement_attempts = int(env_cfg.get("obstacle_placement_attempts", 2000))
        self.obstacle_start_straight_only = bool(env_cfg.get("obstacle_start_straight_only", False))
        self.obstacle_straight_curvature_threshold = float(env_cfg.get("obstacle_straight_curvature_threshold", 0.035))
        self.obstacle_straight_sample_step_m = float(env_cfg.get("obstacle_straight_sample_step_m", 0.5))
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
        self.obstacle_avoidance_lookahead_m = float(
            env_cfg.get("obstacle_avoidance_lookahead_m", max(12.0, self.max_speed * 3.0))
        )
        self.obstacle_clearance_m = float(env_cfg.get("obstacle_clearance_m", self.vehicle_width + 0.5))
        self.wall_clearance_m = float(env_cfg.get("wall_clearance_m", self.obstacle_clearance_m))

        # action = [steer_correction_ratio] for new models. Older saved models may
        # still output [target_speed_ratio, steer_correction_ratio]; speed is fixed.
        self.action_dim = int(env_cfg.get("action_dim", 2))
        self.action_space = spaces.Box(
            low=-np.ones(self.action_dim, dtype=np.float32),
            high=np.ones(self.action_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.ones(8, dtype=np.float32), high=np.ones(8, dtype=np.float32))
        self.state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)
        self.steps = 0
        self.progress_s = 0.0
        self.prev_s = 0.0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.lap_count = 0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.finish_progress_s = self.track.length * self.finish_laps
        self.episode_finished = False
        self.finish_reason = ""
        self.stopped_steps = 0
        self.commanded_steer = 0.0
        self.localized_state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)
        self.localized_prev_s = 0.0
        self._state_history: deque[tuple[float, VehicleState]] = deque()
        self._steer_command_history: deque[tuple[float, float]] = deque()
        self.obstacle_vehicles: list[ObstacleVehicle] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        start_s = (
            float(self.np_random.uniform(0.0, self.track.length))
            if options and options.get("random_start")
            else self.start_s
        )
        x, y, yaw, _ = self.track.sample_at(start_s)
        lateral = float(self.np_random.normal(0.0, self.start_noise_m))
        x -= lateral * np.sin(yaw)
        y += lateral * np.cos(yaw)
        yaw = float(wrap_angle(yaw + self.np_random.normal(0.0, self.start_noise_yaw)))
        self.state = VehicleState(x=x, y=y, yaw=yaw, speed=1.0, steer=0.0)
        self.obstacle_vehicles = self._generate_obstacle_vehicles(start_s)
        self.finish_progress_s = self._finish_progress_for_start(start_s)
        proj = self.track.project(x, y, yaw)
        self.steps = 0
        self.progress_s = 0.0
        self.prev_s = proj.s
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.lap_count = 0
        self.prev_x = self.state.x
        self.prev_y = self.state.y
        self.episode_finished = False
        self.finish_reason = ""
        self.stopped_steps = 0
        self.commanded_steer = 0.0
        self.localized_state = self._copy_state(self.state)
        self.localized_prev_s = proj.s
        self._state_history.clear()
        self._state_history.append((0.0, self._copy_state(self.state)))
        self._steer_command_history.clear()
        self._steer_command_history.append((0.0, self.commanded_steer))
        return self._obs(proj), self._info(proj, collision=False, localized_proj=proj)

    def _resolve_start_s(self) -> float:
        if not self.start_pose_awsim:
            return float(self.config["env"].get("start_s_m", 0.0))

        pose = self.start_pose_awsim
        x = float(pose["x"]) - float(self.track.origin[0])
        y = float(pose["y"]) - float(self.track.origin[1])
        if "yaw_rad" in pose:
            yaw = float(pose["yaw_rad"])
        elif "orientation" in pose:
            orientation = pose["orientation"]
            yaw = 2.0 * np.arctan2(float(orientation["z"]), float(orientation["w"]))
        else:
            yaw = 0.0
        return float(self.track.project(x, y, yaw).s)

    def _finish_progress_for_start(self, start_s: float) -> float:
        if not self.finish_on_start_straight_exit:
            return self.track.length * self.finish_laps
        _, straight_end_s = self._straight_section_around(start_s)
        return max(straight_end_s - start_s, self.vehicle_length)

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
        straight_finished = self.progress_s >= self.finish_progress_s
        stopped = self.stopped_steps >= self.max_stopped_steps
        self.episode_finished = bool((lap_finished or straight_finished) and not collision and not stopped)
        if self.episode_finished:
            self.finish_reason = "straight complete" if straight_finished and not lap_finished else "lap complete"
        terminated = bool(collision or lap_finished or straight_finished or stopped)
        truncated = bool(self.steps >= self.max_episode_steps)
        distance_moved = float(np.hypot(self.state.x - self.prev_x, self.state.y - self.prev_y))
        reward = self._reward(proj, progress, action, collision, self.episode_finished, stopped, distance_moved)
        self.prev_action = action.copy()
        self.prev_x = self.state.x
        self.prev_y = self.state.y
        return self._obs(localized_proj), reward, terminated, truncated, self._info(
            proj,
            collision=collision,
            localized_proj=localized_proj,
        )

    def _is_collision(self, proj) -> bool:
        margin = self.boundary_margin
        wall_collision = proj.lateral_error < proj.lateral_min + margin or proj.lateral_error > proj.lateral_max - margin
        return wall_collision or self._collides_with_obstacle()

    def _resolve_collision(self, proj):
        if self._collides_with_obstacle():
            return proj
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
        target_speed = self.max_speed
        speed_error = target_speed - self.state.speed
        accel_limit = self.max_accel if speed_error >= 0.0 else self.max_brake
        speed_step = float(np.clip(speed_error, -accel_limit * self.dt, accel_limit * self.dt))
        self.state.speed = float(np.clip(self.state.speed + speed_step, self.min_speed, self.max_speed))
        base_steer = self._pure_pursuit_steer(self.localized_prev_s)
        steer_correction = float(action[-1]) * self.max_steer * self.max_steer_correction_ratio
        self._set_commanded_steer(base_steer + steer_correction)

    def _generate_obstacle_vehicles(self, start_s: float) -> list[ObstacleVehicle]:
        if self.obstacle_vehicle_count <= 0:
            return []

        vehicles: list[ObstacleVehicle] = []
        min_half_width = 0.5 * self.vehicle_width + self.boundary_margin + self.obstacle_lateral_margin_m
        min_gap = max(self.obstacle_min_gap_m, self.vehicle_length * 2.0)
        start_clearance = max(self.obstacle_start_clearance_m, min_gap)
        s_ranges = self._obstacle_s_ranges(start_s) if self.obstacle_start_straight_only else None
        if s_ranges is not None:
            return self._generate_obstacle_vehicles_in_ranges(s_ranges, start_s, min_half_width, min_gap, start_clearance)

        for _ in range(self.obstacle_placement_attempts):
            if len(vehicles) >= self.obstacle_vehicle_count:
                break
            s = self._sample_obstacle_s(None)
            if self._track_gap(s, start_s) < start_clearance:
                continue
            if any(self._track_gap(s, vehicle.s) < min_gap for vehicle in vehicles):
                continue

            center_x, center_y, yaw, _ = self.track.sample_at(s)
            proj = self.track.project(center_x, center_y, yaw)
            lateral_min = proj.lateral_min + min_half_width
            lateral_max = proj.lateral_max - min_half_width
            if lateral_min > lateral_max:
                continue

            placement = self._place_obstacle_near_wall(s, center_x, center_y, yaw, lateral_min, lateral_max, vehicles)
            if placement is None:
                continue
            candidate, _ = placement
            vehicles.append(candidate)

        if len(vehicles) != self.obstacle_vehicle_count:
            raise RuntimeError(f"Could only place {len(vehicles)} obstacle vehicles out of {self.obstacle_vehicle_count}")
        return vehicles

    def _generate_obstacle_vehicles_in_ranges(
        self,
        s_ranges: list[tuple[float, float]],
        start_s: float,
        min_half_width: float,
        min_gap: float,
        start_clearance: float,
    ) -> list[ObstacleVehicle]:
        for _ in range(self.obstacle_placement_attempts):
            vehicles: list[ObstacleVehicle] = []
            for s in self._sample_obstacle_s_values(s_ranges, start_s, start_clearance, min_gap):
                center_x, center_y, yaw, _ = self.track.sample_at(s)
                proj = self.track.project(center_x, center_y, yaw)
                lateral_min = proj.lateral_min + min_half_width
                lateral_max = proj.lateral_max - min_half_width
                if lateral_min > lateral_max:
                    break

                placement = self._place_obstacle_near_wall(s, center_x, center_y, yaw, lateral_min, lateral_max, vehicles)
                if placement is None:
                    break
                candidate, _ = placement
                vehicles.append(candidate)

            if len(vehicles) == self.obstacle_vehicle_count:
                return vehicles

        raise RuntimeError(
            f"Could only place obstacles in the start straight section after {self.obstacle_placement_attempts} attempts"
        )

    def _obstacle_s_ranges(self, start_s: float) -> list[tuple[float, float]]:
        section_start, section_end = self._straight_section_around(start_s)
        return [(section_start, section_end)]

    def _straight_section_around(self, start_s: float) -> tuple[float, float]:
        step = max(self.obstacle_straight_sample_step_m, 0.05)
        threshold = max(self.obstacle_straight_curvature_threshold, 0.0)

        backward = 0.0
        while backward + step < self.track.length:
            _, _, _, curvature = self.track.sample_at(start_s - backward - step)
            if abs(curvature) > threshold:
                break
            backward += step

        forward = 0.0
        while forward + step < self.track.length - backward:
            _, _, _, curvature = self.track.sample_at(start_s + forward + step)
            if abs(curvature) > threshold:
                break
            forward += step

        if backward + forward < self.obstacle_min_gap_m:
            raise RuntimeError("Could not find a long enough straight section around the start position")
        return start_s - backward, start_s + forward

    def _sample_obstacle_s(self, s_ranges: list[tuple[float, float]] | None) -> float:
        if not s_ranges:
            return float(self.np_random.uniform(0.0, self.track.length))

        lengths = np.array([max(end - start, 0.0) for start, end in s_ranges], dtype=np.float64)
        total = float(lengths.sum())
        if total <= 0.0:
            raise RuntimeError("Obstacle placement range is empty")
        selected = int(self.np_random.choice(len(s_ranges), p=lengths / total))
        start, end = s_ranges[selected]
        return float(self.np_random.uniform(start, end) % self.track.length)

    def _sample_obstacle_s_values(
        self,
        s_ranges: list[tuple[float, float]],
        start_s: float,
        start_clearance: float,
        min_gap: float,
    ) -> list[float]:
        intervals: list[tuple[float, float]] = []
        for range_start, range_end in s_ranges:
            intervals.extend(
                [
                    (range_start, min(range_end, start_s - start_clearance)),
                    (max(range_start, start_s + start_clearance), range_end),
                ]
            )

        intervals = [(start, end) for start, end in intervals if end - start >= 0.0]
        self.np_random.shuffle(intervals)
        count = self.obstacle_vehicle_count
        for start, end in intervals:
            free_length = end - start - min_gap * (count - 1)
            if free_length < 0.0:
                continue
            offsets = np.sort(self.np_random.uniform(0.0, free_length, count))
            return [float((start + offsets[idx] + idx * min_gap) % self.track.length) for idx in range(count)]

        raise RuntimeError("Could not fit obstacle vehicles in the start straight section")

    def _place_obstacle_near_wall(
        self,
        s: float,
        center_x: float,
        center_y: float,
        yaw: float,
        lateral_min: float,
        lateral_max: float,
        vehicles: list[ObstacleVehicle],
    ) -> tuple[ObstacleVehicle, np.ndarray] | None:
        side_indices = self.np_random.permutation(2)
        max_inward_offset = max(lateral_max - lateral_min, 0.0)
        step = 0.05
        normal = np.array([-np.sin(yaw), np.cos(yaw)])

        for side_idx in side_indices:
            edge_lateral = lateral_min if side_idx == 0 else lateral_max
            inward_direction = 1.0 if side_idx == 0 else -1.0
            for inward_offset in np.arange(0.0, max_inward_offset + step, step):
                lateral = float(edge_lateral + inward_direction * inward_offset)
                if lateral < lateral_min or lateral > lateral_max:
                    continue
                x = float(center_x + normal[0] * lateral)
                y = float(center_y + normal[1] * lateral)
                candidate = ObstacleVehicle(s=s, x=x, y=y, yaw=float(yaw), lateral=lateral)
                candidate_polygon = self._vehicle_polygon_at(candidate.x, candidate.y, candidate.yaw)
                if self._intersects_lane_boundary(candidate_polygon):
                    continue
                if any(
                    _polygons_intersect(candidate_polygon, self._vehicle_polygon_at(vehicle.x, vehicle.y, vehicle.yaw))
                    for vehicle in vehicles
                ):
                    continue
                return candidate, candidate_polygon
        return None

    def _collides_with_obstacle(self) -> bool:
        ego_polygon = self._vehicle_polygon()
        return any(_polygons_intersect(ego_polygon, self._vehicle_polygon_at(vehicle.x, vehicle.y, vehicle.yaw)) for vehicle in self.obstacle_vehicles)

    def _vehicle_polygon(self) -> np.ndarray:
        return self._vehicle_polygon_at(self.state.x, self.state.y, self.state.yaw)

    def _vehicle_polygon_at(self, x: float, y: float, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        half_l = 0.5 * self.vehicle_length
        half_w = 0.5 * self.vehicle_width
        corners = np.array(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ],
            dtype=np.float64,
        )
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        return corners @ rot.T + np.array([x, y], dtype=np.float64)

    def _intersects_lane_boundary(self, polygon: np.ndarray) -> bool:
        if self.track.lane_segments is None or len(self.track.lane_segments) == 0:
            return False
        center = polygon.mean(axis=0)
        radius = 0.5 * np.hypot(self.vehicle_length, self.vehicle_width)
        p0 = self.track.lane_segments[:, 0, :]
        p1 = self.track.lane_segments[:, 1, :]
        midpoint = p0 + 0.5 * (p1 - p0)
        nearby = np.linalg.norm(midpoint - center, axis=1) <= radius + 0.5 * np.linalg.norm(p1 - p0, axis=1)
        return any(_segment_intersects_polygon(a, b, polygon) for a, b in self.track.lane_segments[nearby])

    def _obstacle_segments(self) -> np.ndarray:
        if not self.obstacle_vehicles:
            return np.empty((0, 2, 2), dtype=np.float64)
        segments = []
        for vehicle in self.obstacle_vehicles:
            polygon = self._vehicle_polygon_at(vehicle.x, vehicle.y, vehicle.yaw)
            segments.append(np.stack([polygon, np.roll(polygon, -1, axis=0)], axis=1))
        return np.concatenate(segments, axis=0)

    def _track_gap(self, a: float, b: float) -> float:
        return float(abs((a - b + 0.5 * self.track.length) % self.track.length - 0.5 * self.track.length))

    def _nearest_forward_obstacle(self, s: float, lookahead_m: float | None = None) -> tuple[ObstacleVehicle | None, float]:
        lookahead = self.obstacle_avoidance_lookahead_m if lookahead_m is None else lookahead_m
        nearest: ObstacleVehicle | None = None
        nearest_delta = float("inf")
        for vehicle in self.obstacle_vehicles:
            delta = float((vehicle.s - s) % self.track.length)
            if 1e-6 < delta <= lookahead and delta < nearest_delta:
                nearest = vehicle
                nearest_delta = delta
        return nearest, nearest_delta

    def _obstacle_avoidance_penalty(self, proj) -> float:
        obstacle, delta_s = self._nearest_forward_obstacle(proj.s)
        if obstacle is None:
            return 0.0
        proximity = 1.0 - np.clip(delta_s / max(self.obstacle_avoidance_lookahead_m, 1e-6), 0.0, 1.0)
        lateral_clearance = abs(proj.lateral_error - obstacle.lateral)
        clearance_deficit = np.clip(
            (self.obstacle_clearance_m - lateral_clearance) / max(self.obstacle_clearance_m, 1e-6),
            0.0,
            1.0,
        )
        return float((proximity * proximity) * clearance_deficit)

    def _wall_avoidance_penalty(self, proj) -> float:
        wall_clearance = min(proj.lateral_error - proj.lateral_min, proj.lateral_max - proj.lateral_error)
        clearance_deficit = np.clip(
            (self.wall_clearance_m - wall_clearance) / max(self.wall_clearance_m, 1e-6),
            0.0,
            1.0,
        )
        return float(clearance_deficit)

    def _hazard_avoidance_penalty(self, proj) -> float:
        return max(self._wall_avoidance_penalty(proj), self._obstacle_avoidance_penalty(proj))

    def obstacle_state_features(self, proj) -> np.ndarray:
        obstacle, delta_s = self._nearest_forward_obstacle(proj.s)
        features = np.zeros(3, dtype=np.float32)
        if obstacle is None:
            features[0] = 1.0
            return features

        local_half_width = max(abs(proj.lateral_min), abs(proj.lateral_max), 1e-3)
        features[0] = np.clip(delta_s / max(self.obstacle_avoidance_lookahead_m, 1e-6), 0.0, 1.0)
        features[1] = 0.5 * (np.clip(obstacle.lateral / local_half_width, -1.0, 1.0) + 1.0)
        features[2] = np.clip(abs(proj.lateral_error - obstacle.lateral) / max(self.obstacle_clearance_m, 1e-6), 0.0, 1.0)
        return features

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
        target_steer = self._delayed_steer(time_sec - self.steering_delay_sec)
        max_delta = self.max_steer_rate * self.dt
        steer_delta = float(np.clip(target_steer - self.state.steer, -max_delta, max_delta))
        self.state.steer = float(np.clip(self.state.steer + steer_delta, -self.max_steer, self.max_steer))

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
        finished: bool,
        stopped: bool,
        distance_moved: float,
    ) -> float:
        r = self.reward_cfg
        reward = r["progress"] * progress
        reward += r["speed"] * self.state.speed
        if distance_moved > 1e-6:
            reward -= r.get("wasted_motion", 0.0) * max(distance_moved - max(progress, 0.0), 0.0)
        lateral_error_weight = r.get("lateral_error", 0.0)
        if lateral_error_weight:
            local_half_width = max(abs(proj.lateral_min), abs(proj.lateral_max), 1e-3)
            reward -= lateral_error_weight * abs(proj.lateral_error / local_half_width)
        reward -= r["heading_error"] * abs(proj.heading_error / np.pi)
        reward -= r["steer"] * abs(self.state.steer / self.max_steer)
        reward -= r["action_smooth"] * float(np.linalg.norm(action - self.prev_action))
        if self.state.speed < self.min_moving_speed:
            reward -= r.get("low_speed", 0.0)
        hazard_avoidance_weight = r.get("hazard_avoidance", r.get("obstacle_avoidance", 0.0))
        reward -= hazard_avoidance_weight * self._hazard_avoidance_penalty(proj)
        if collision:
            reward -= r["wall_collision"]
        if stopped:
            reward -= r.get("stopped", 0.0)
        if finished:
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
            "finished": self.episode_finished,
            "finish_reason": self.finish_reason,
            "finish_progress_s": self.finish_progress_s,
            "time": self.steps * self.dt,
        }

    def render(self):
        return None


def _polygons_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    for polygon in (a, b):
        for idx in range(len(polygon)):
            edge = polygon[(idx + 1) % len(polygon)] - polygon[idx]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                continue
            axis /= norm
            a_proj = a @ axis
            b_proj = b @ axis
            if a_proj.max() < b_proj.min() or b_proj.max() < a_proj.min():
                return False
    return True


def _segment_intersects_polygon(a: np.ndarray, b: np.ndarray, polygon: np.ndarray) -> bool:
    if _point_in_polygon(a, polygon) or _point_in_polygon(b, polygon):
        return True
    for idx in range(len(polygon)):
        c = polygon[idx]
        d = polygon[(idx + 1) % len(polygon)]
        if _segments_intersect(a, b, c, d):
            return True
    return False


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    ab = b - a
    cd = d - c
    ac = c - a
    denom = _cross(ab, cd)
    if abs(float(denom)) < 1e-9:
        return _point_on_segment(c, a, b) or _point_on_segment(d, a, b) or _point_on_segment(a, c, d) or _point_on_segment(b, c, d)
    t = _cross(ac, cd) / denom
    u = _cross(ac, ab) / denom
    return bool(0.0 <= t <= 1.0 and 0.0 <= u <= 1.0)


def _point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    return bool(abs(float(_cross(b - a, p - a))) < 1e-9 and np.dot(p - a, p - b) <= 1e-9)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
