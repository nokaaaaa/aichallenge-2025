from __future__ import annotations

import argparse
import copy
import math
import re
from pathlib import Path

import numpy as np
import pygame

from kart_rl.config import PACKAGE_ROOT, load_config, resolve_path
from kart_rl.evaluate import collect_rollout_data, resolve_model_path


def world_to_screen(points, bounds, screen_size, margin=48):
    min_xy, max_xy = bounds
    span = np.maximum(max_xy - min_xy, 1.0)
    scale = min((screen_size[0] - 2 * margin) / span[0], (screen_size[1] - 2 * margin) / span[1])
    screen = (points - min_xy) * scale + margin
    screen[:, 1] = screen_size[1] - screen[:, 1]
    return screen


def draw_vehicle(surface, xy, yaw, length, width, bounds, screen_size, collision=False, fill=(238, 183, 58)):
    center = world_to_screen(np.array([xy], dtype=np.float32), bounds, screen_size)[0]
    min_xy, max_xy = bounds
    scale = min((screen_size[0] - 96) / max(max_xy[0] - min_xy[0], 1.0), (screen_size[1] - 96) / max(max_xy[1] - min_xy[1], 1.0))
    l = max(length * scale, 12.0)
    w = max(width * scale, 8.0)
    c, s = math.cos(-yaw), math.sin(-yaw)
    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    rot = np.array([[c, -s], [s, c]])
    pts = corners @ rot.T + center
    pygame.draw.polygon(surface, fill, pts)
    outline = (196, 36, 32) if collision else (77, 65, 32)
    pygame.draw.polygon(surface, outline, pts, width=3 if collision else 2)


def vehicle_segments(vehicles: np.ndarray, length: float, width: float) -> np.ndarray:
    if len(vehicles) == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    half_l = 0.5 * length
    half_w = 0.5 * width
    corners = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]], dtype=np.float64)
    segments = []
    for x, y, yaw in vehicles:
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        polygon = corners @ rot.T + np.array([x, y], dtype=np.float64)
        segments.append(np.stack([polygon, np.roll(polygon, -1, axis=0)], axis=1))
    return np.concatenate(segments, axis=0)


def termination_reason(frame, frames, config):
    if bool(frame[9]):
        return "collision"
    if bool(frame[12]):
        return "stopped"
    if frame[7] >= 0.999:
        return "lap complete"
    timeout_time = float(config["env"]["max_episode_steps"]) * float(config["env"]["dt"])
    if abs(float(frame[0]) - timeout_time) <= float(config["env"]["dt"]) * 1.5:
        return "timeout"
    if len(frames) > 0:
        return "log end"
    return "unknown"


def draw_lidar(surface, frame, ranges, angles, range_max, vehicle_length, bounds, screen_size):
    if ranges is None or angles is None or len(ranges) == 0:
        return

    yaw = float(frame[3])
    origin = np.array(
        [
            frame[1] + 0.5 * vehicle_length * math.cos(yaw),
            frame[2] + 0.5 * vehicle_length * math.sin(yaw),
        ],
        dtype=np.float32,
    )
    ray_ranges = np.asarray(ranges, dtype=np.float32)
    ray_angles = yaw + np.asarray(angles, dtype=np.float32)
    endpoints = origin + np.column_stack([np.cos(ray_angles), np.sin(ray_angles)]) * ray_ranges[:, None]

    origin_screen = world_to_screen(np.array([origin], dtype=np.float32), bounds, screen_size)[0].astype(int)
    endpoint_screen = world_to_screen(endpoints.astype(np.float32), bounds, screen_size).astype(int)
    overlay = pygame.Surface(screen_size, pygame.SRCALPHA)
    for distance, endpoint in zip(ray_ranges, endpoint_screen):
        color = (46, 160, 180, 34) if distance >= range_max * 0.999 else (17, 134, 157, 72)
        pygame.draw.line(overlay, color, origin_screen, endpoint, width=1)
        if distance < range_max * 0.999:
            pygame.draw.circle(overlay, (10, 92, 112, 120), endpoint, 2)
    pygame.draw.circle(overlay, (10, 92, 112, 180), origin_screen, 4)
    surface.blit(overlay, (0, 0))


def lane_edge_polylines(lane_segments: np.ndarray) -> list[np.ndarray]:
    if len(lane_segments) == 0:
        return []
    edges = [lane_segments[:, 0, :], lane_segments[:, 1, :]]
    polylines = []
    for edge in edges:
        finite = edge[np.isfinite(edge).all(axis=1)]
        if len(finite) < 2:
            continue
        step = np.linalg.norm(np.diff(finite, axis=0), axis=1)
        keep = np.concatenate([[True], step > 1e-6])
        polylines.append(finite[keep])
    return polylines


def lane_edge_polylines_from_config(config: dict) -> list[np.ndarray]:
    track_cfg = config.get("track", {})
    lane_csv_path = track_cfg.get("lane_csv_path")
    track_csv_path = track_cfg.get("csv_path")
    if not lane_csv_path or not track_csv_path:
        return []

    track_data = np.genfromtxt(resolve_path(track_csv_path, config, must_exist=True), delimiter=",", names=True)
    origin = np.array([track_data["x"][0], track_data["y"][0]], dtype=np.float64)
    lane_data = np.genfromtxt(resolve_path(lane_csv_path, config, must_exist=True), delimiter=",")

    polylines = []
    for cols in ((0, 1), (2, 3)):
        points = lane_data[:, cols]
        points = points[np.isfinite(points).all(axis=1)] - origin
        if len(points) < 2:
            continue
        step = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.concatenate([[True], step > 1e-6])
        polylines.append(points[keep])
    return polylines


def lidar_angles_from_config(config: dict) -> np.ndarray | None:
    lidar_cfg = config.get("lidar")
    if not lidar_cfg:
        return None
    angle_min = float(lidar_cfg["angle_min"])
    angle_max = float(lidar_cfg["angle_max"])
    angle_increment = float(lidar_cfg["angle_increment"])
    sample_ratio = float(lidar_cfg.get("sample_ratio", 1.0))
    stride = max(1, int(round(1.0 / max(sample_ratio, 1e-3))))
    return np.arange(angle_min, angle_max + 0.5 * angle_increment, angle_increment, dtype=np.float32)[::stride]


def scan_lidar_for_frame(
    frame: np.ndarray,
    lane_segments: np.ndarray,
    obstacle_segments: np.ndarray,
    angles: np.ndarray,
    range_max: float,
    vehicle_length: float,
) -> np.ndarray:
    yaw = float(frame[3])
    origin = np.array(
        [
            frame[1] + 0.5 * vehicle_length * math.cos(yaw),
            frame[2] + 0.5 * vehicle_length * math.sin(yaw),
        ],
        dtype=np.float64,
    )
    if len(lane_segments) == 0 and len(obstacle_segments) == 0:
        return np.full(len(angles), range_max, dtype=np.float32)

    scan_segments = lane_segments
    if len(obstacle_segments) > 0:
        scan_segments = np.concatenate([scan_segments, obstacle_segments], axis=0) if len(scan_segments) else obstacle_segments

    lane_p0 = scan_segments[:, 0, :]
    lane_v = scan_segments[:, 1, :] - scan_segments[:, 0, :]
    midpoint = lane_p0 + 0.5 * lane_v
    seg_radius = 0.5 * np.linalg.norm(lane_v, axis=1)
    nearby = np.linalg.norm(midpoint - origin, axis=1) <= range_max + seg_radius
    lane_p0 = lane_p0[nearby]
    lane_v = lane_v[nearby]
    if len(lane_p0) == 0:
        return np.full(len(angles), range_max, dtype=np.float32)

    ray_angles = yaw + angles
    rays = np.column_stack([np.cos(ray_angles), np.sin(ray_angles)])
    rel = lane_p0 - origin
    ranges = np.full(len(angles), range_max, dtype=np.float64)
    for start in range(0, len(rays), 64):
        ray_chunk = rays[start : start + 64]
        hit = ray_segment_distances_batch(rel, ray_chunk, lane_v)
        ranges[start : start + len(ray_chunk)] = np.minimum(ranges[start : start + len(ray_chunk)], hit)
    return ranges.astype(np.float32)


def ray_segment_distances_batch(rel: np.ndarray, rays: np.ndarray, seg_v: np.ndarray) -> np.ndarray:
    denom = cross(rays[:, None, :], seg_v[None, :, :])
    valid = np.abs(denom) > 1e-9
    t = np.full_like(denom, np.inf, dtype=np.float64)
    u = np.full_like(denom, np.inf, dtype=np.float64)
    np.divide(cross(rel[None, :, :], seg_v[None, :, :]), denom, out=t, where=valid)
    np.divide(cross(rel[None, :, :], rays[:, None, :]), denom, out=u, where=valid)
    hit = valid & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    return np.clip(np.where(hit, t, np.inf).min(axis=1), 0.0, np.inf)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def read_npz_string(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    return str(value.item() if value.shape == () else value)


def display_model_name(model_path: Path | None, fallback: str = "unknown") -> str:
    if model_path is None:
        return fallback
    parent = model_path.parent.name
    return parent if parent != "models" else model_path.stem


def load_config_for_model(model_path: Path, base_config: dict) -> dict:
    config_path = model_path.parent / "config.yaml"
    if not config_path.exists():
        return copy.deepcopy(base_config)
    model_config = load_config(config_path)
    config = copy.deepcopy(model_config)
    for key, value in base_config.items():
        if key not in config:
            config[key] = copy.deepcopy(value)
    env_cfg = config.setdefault("env", {})
    env_cfg.setdefault("obstacle_vehicle_count", 0)
    env_cfg.setdefault("localization_delay_sec", 0.0)
    env_cfg.setdefault("steering_delay_sec", 0.0)
    # Saved training configs were copied from configs/default.yaml, so keep relative paths
    # such as lane.csv anchored to the viewer's base config directory.
    config["_config_path"] = base_config["_config_path"]
    return config


def discover_model_paths(config: dict, selected_model: Path | None) -> list[Path]:
    model_setting = config["viewer"].get("model_path") or config["train"]["model_path"]
    base = resolve_path(model_setting, config)
    stem = base.with_suffix("") if base.suffix else base
    model_dir = stem.parent
    run_dir_pattern = re.compile(rf"^{re.escape(stem.name)}_\d{{8}}-\d{{6}}(?:_\d+)?$")
    candidates: set[Path] = set()
    if model_dir.exists():
        for path in (model_dir / f"{stem.name}.zip", model_dir / f"{stem.name}_latest.zip"):
            if path.exists():
                candidates.add(path.resolve())
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir() or not run_dir_pattern.fullmatch(run_dir.name):
                continue
            path = run_dir / f"{stem.name}.zip"
            if path.exists():
                candidates.add(path.resolve())
    if selected_model is not None:
        candidates.add(selected_model.resolve())
    return sorted(candidates, key=lambda path: (path.parent.name, path.name))


def prepare_view_data(data: np.lib.npyio.NpzFile | dict, config: dict, screen_size: tuple[int, int]) -> dict:
    frames = data["frames"]
    track = data["track"]
    left_boundary = data["left_boundary"] if "left_boundary" in data else np.empty((0, 2), dtype=np.float32)
    right_boundary = data["right_boundary"] if "right_boundary" in data else np.empty((0, 2), dtype=np.float32)
    lane_segments = data["lane_segments"] if "lane_segments" in data else np.empty((0, 2, 2), dtype=np.float32)
    lane_edges = lane_edge_polylines_from_config(config) or lane_edge_polylines(lane_segments)
    vehicle_length = float(data["vehicle_length"])
    vehicle_width = float(data["vehicle_width"])
    obstacle_vehicles = data["obstacle_vehicles"] if "obstacle_vehicles" in data else np.empty((0, 3), dtype=np.float32)
    obstacle_segments = vehicle_segments(obstacle_vehicles, vehicle_length, vehicle_width)
    lidar_ranges = data["lidar_ranges"] if "lidar_ranges" in data else None
    lidar_angles = data["lidar_angles"] if "lidar_angles" in data else lidar_angles_from_config(config)
    lidar_range_max = float(data["lidar_range_max"]) if "lidar_range_max" in data else float(config.get("lidar", {}).get("range_max", 25.0))
    if lidar_ranges is not None and lidar_angles is not None:
        lidar_ranges = lidar_ranges[:, : len(lidar_angles)]

    bound_points = [track, frames[:, 1:3]]
    if len(obstacle_vehicles):
        bound_points.append(obstacle_vehicles[:, 0:2])
    if len(left_boundary) and len(right_boundary):
        bound_points.extend([left_boundary, right_boundary])
    bound_points.extend(lane_edges)
    all_points = np.vstack(bound_points)
    bounds = (all_points.min(axis=0) - 4.0, all_points.max(axis=0) + 4.0)
    driven_line = world_to_screen(frames[:, 1:3], bounds, screen_size).astype(int)

    return {
        "frames": frames,
        "track": track,
        "lane_segments": lane_segments,
        "lane_edges": lane_edges,
        "vehicle_length": vehicle_length,
        "vehicle_width": vehicle_width,
        "obstacle_vehicles": obstacle_vehicles,
        "obstacle_segments": obstacle_segments,
        "lidar_ranges": lidar_ranges,
        "lidar_angles": lidar_angles,
        "lidar_range_max": lidar_range_max,
        "bounds": bounds,
        "driven_line": driven_line,
        "computed_lidar_cache": {},
    }


def draw_button(surface, font, rect: pygame.Rect, label: str, enabled: bool = True) -> None:
    bg = (248, 249, 246) if enabled else (214, 216, 212)
    border = (86, 92, 94) if enabled else (150, 154, 151)
    text_color = (25, 29, 33) if enabled else (112, 116, 112)
    pygame.draw.rect(surface, bg, rect, border_radius=4)
    pygame.draw.rect(surface, border, rect, width=1, border_radius=4)
    text = font.render(label, True, text_color)
    surface.blit(text, text.get_rect(center=rect.center))


def env_summary(config: dict) -> str:
    env_cfg = config.get("env", {})
    obstacles = int(env_cfg.get("obstacle_vehicle_count", 0))
    localization_delay = float(env_cfg.get("localization_delay_sec", 0.0))
    steering_delay = float(env_cfg.get("steering_delay_sec", 0.0))
    return f"env: obstacles={obstacles}  loc_delay={localization_delay:.2f}s  steer_delay={steering_delay:.2f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--rollout", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    active_config = config
    selected_model_path: Path | None = None
    model_error: str | None = None
    total_reward = 0.0
    info = {"time": 0.0, "lap_count": 0, "collision": False}
    model_paths: list[Path] = []
    selected_model_index = -1
    if args.rollout:
        data = np.load(resolve_path(args.rollout, config, must_exist=True))
        model_dir_name = read_npz_string(data, "model_dir_name", "unknown")
    else:
        model_setting = args.model or config["viewer"].get("model_path") or config["train"]["model_path"]
        selected_model_path = resolve_model_path(model_setting, config)
        model_paths = discover_model_paths(config, selected_model_path)
        for index, path in enumerate(model_paths):
            if path == selected_model_path.resolve():
                selected_model_index = index
                break
        load_order = [selected_model_index]
        load_order.extend(range(selected_model_index - 1, -1, -1))
        load_order.extend(range(selected_model_index + 1, len(model_paths)))
        data = None
        errors = []
        for index in load_order:
            path = model_paths[index]
            candidate_config = load_config_for_model(path, config)
            try:
                data, total_reward, info = collect_rollout_data(candidate_config, path, args.deterministic)
            except Exception as exc:
                errors.append(f"{display_model_name(path)}: {exc}")
                print(f"Failed to load model: {path}: {exc}")
                continue
            active_config = candidate_config
            selected_model_index = index
            selected_model_path = path
            model_dir_name = display_model_name(path)
            print(f"Loaded model: {path}")
            print(f"Loaded config: {path.parent / 'config.yaml' if (path.parent / 'config.yaml').exists() else args.config}")
            print(f"reward={total_reward:.2f} time={info['time']:.2f}s laps={info['lap_count']} collision={info['collision']}")
            break
        if data is None:
            raise RuntimeError("No loadable model found:\n" + "\n".join(errors))

    pygame.init()
    screen_size = (1100, 800)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Kart RL Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)
    small_font = pygame.font.SysFont("monospace", 16)

    view = prepare_view_data(data, active_config, screen_size)
    show_lidar = view["lidar_angles"] is not None
    if not model_paths:
        model_paths = discover_model_paths(config, selected_model_path)
    if selected_model_path is not None:
        selected_resolved = selected_model_path.resolve()
        for index, path in enumerate(model_paths):
            if path == selected_resolved:
                selected_model_index = index
                break

    def select_model(index: int) -> None:
        nonlocal active_config, data, info, idx, model_dir_name, model_error, paused, selected_model_index, selected_model_path, show_lidar, total_reward, view
        if not model_paths:
            return
        previous_config = active_config
        previous_model_index = selected_model_index
        previous_model_path = selected_model_path
        previous_model_dir_name = model_dir_name
        selected_model_index = index % len(model_paths)
        selected_model_path = model_paths[selected_model_index]
        model_dir_name = display_model_name(selected_model_path)
        candidate_config = load_config_for_model(selected_model_path, config)
        paused = True
        model_error = None
        screen.fill((236, 238, 232))
        screen.blit(font.render(f"loading model: {model_dir_name}", True, (25, 29, 33)), (18, 16))
        pygame.display.flip()
        try:
            data, total_reward, info = collect_rollout_data(candidate_config, selected_model_path, args.deterministic)
            active_config = candidate_config
            view = prepare_view_data(data, active_config, screen_size)
            show_lidar = view["lidar_angles"] is not None
            idx = 0
            print(f"Loaded model: {selected_model_path}")
            print(
                f"Loaded config: {selected_model_path.parent / 'config.yaml' if (selected_model_path.parent / 'config.yaml').exists() else args.config}"
            )
            print(f"reward={total_reward:.2f} time={info['time']:.2f}s laps={info['lap_count']} collision={info['collision']}")
        except Exception as exc:
            active_config = previous_config
            selected_model_index = previous_model_index
            selected_model_path = previous_model_path
            model_dir_name = previous_model_dir_name
            model_error = str(exc)

    idx = 0
    paused = False
    running = True
    fps = int(config["viewer"].get("fps", 50))
    prev_button = pygame.Rect(screen_size[0] - 196, 15, 42, 30)
    next_button = pygame.Rect(screen_size[0] - 146, 15, 42, 30)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if prev_button.collidepoint(event.pos) and model_paths:
                    select_model(selected_model_index - 1)
                elif next_button.collidepoint(event.pos) and model_paths:
                    select_model(selected_model_index + 1)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    idx = 0
                elif event.key == pygame.K_l:
                    show_lidar = not show_lidar
                elif event.key == pygame.K_COMMA and model_paths:
                    select_model(selected_model_index - 1)
                elif event.key == pygame.K_PERIOD and model_paths:
                    select_model(selected_model_index + 1)
                elif event.key == pygame.K_RIGHT:
                    idx = min(idx + fps, len(view["frames"]) - 1)
                elif event.key == pygame.K_LEFT:
                    idx = max(idx - fps, 0)

        if not paused:
            idx = min(idx + 1, len(view["frames"]) - 1)

        screen.fill((236, 238, 232))
        for edge in view["lane_edges"]:
            pts = world_to_screen(edge, view["bounds"], screen_size).astype(int)
            pygame.draw.lines(screen, (116, 120, 114), True, pts, width=1)
        if idx > 1:
            pygame.draw.lines(screen, (198, 58, 42), False, view["driven_line"][: idx + 1], width=2)

        frame = view["frames"][idx]
        if show_lidar and view["lidar_angles"] is not None:
            if view["lidar_ranges"] is not None:
                current_lidar = view["lidar_ranges"][idx]
            else:
                current_lidar = view["computed_lidar_cache"].get(idx)
                if current_lidar is None:
                    current_lidar = scan_lidar_for_frame(
                        frame,
                        view["lane_segments"],
                        view["obstacle_segments"],
                        view["lidar_angles"],
                        view["lidar_range_max"],
                        view["vehicle_length"],
                    )
                    view["computed_lidar_cache"][idx] = current_lidar
            draw_lidar(
                screen,
                frame,
                current_lidar,
                view["lidar_angles"],
                view["lidar_range_max"],
                view["vehicle_length"],
                view["bounds"],
                screen_size,
            )
        for obstacle in view["obstacle_vehicles"]:
            draw_vehicle(
                screen,
                obstacle[0:2],
                float(obstacle[2]),
                view["vehicle_length"],
                view["vehicle_width"],
                view["bounds"],
                screen_size,
                fill=(80, 105, 130),
            )
        draw_vehicle(screen, frame[1:3], float(frame[3]), view["vehicle_length"], view["vehicle_width"], view["bounds"], screen_size, collision=bool(frame[9]))
        text = f"t={frame[0]:6.2f}s  v={frame[4]:4.2f}m/s  progress={frame[7] * 100:5.1f}%  lateral={frame[8]:+.2f}m"
        screen.blit(font.render(text, True, (25, 29, 33)), (18, 16))
        screen.blit(font.render(f"model: {model_dir_name}", True, (25, 29, 33)), (18, 42))
        screen.blit(font.render(env_summary(active_config), True, (25, 29, 33)), (18, 68))
        screen.blit(font.render("space: pause  left/right: seek  r: restart  l: lidar  </>: model", True, (72, 76, 80)), (18, 94))
        draw_button(screen, small_font, prev_button, "<", bool(model_paths))
        draw_button(screen, small_font, next_button, ">", bool(model_paths))
        model_count = f"{selected_model_index + 1}/{len(model_paths)}" if selected_model_index >= 0 else f"-/{len(model_paths)}"
        screen.blit(small_font.render(model_count, True, (25, 29, 33)), (screen_size[0] - 98, 21))
        status_y = 120
        if model_error:
            screen.blit(font.render(f"model load error: {model_error[:88]}", True, (170, 40, 40)), (18, status_y))
            status_y += 26
        if bool(frame[9]):
            screen.blit(font.render("collision", True, (170, 40, 40)), (18, status_y))
            status_y += 26
        if idx == len(view["frames"]) - 1:
            screen.blit(
                font.render(f"rollout end: {termination_reason(frame, view['frames'], active_config)}", True, (170, 40, 40)),
                (18, status_y),
            )

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()
