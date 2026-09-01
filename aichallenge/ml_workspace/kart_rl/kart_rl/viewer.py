from __future__ import annotations

import argparse
import math

import numpy as np
import pygame

from kart_rl.config import PACKAGE_ROOT, load_config, resolve_path


def world_to_screen(points, bounds, screen_size, margin=48):
    min_xy, max_xy = bounds
    span = np.maximum(max_xy - min_xy, 1.0)
    scale = min((screen_size[0] - 2 * margin) / span[0], (screen_size[1] - 2 * margin) / span[1])
    screen = (points - min_xy) * scale + margin
    screen[:, 1] = screen_size[1] - screen[:, 1]
    return screen


def draw_vehicle(surface, xy, yaw, length, width, bounds, screen_size):
    center = world_to_screen(np.array([xy], dtype=np.float32), bounds, screen_size)[0]
    min_xy, max_xy = bounds
    scale = min((screen_size[0] - 96) / max(max_xy[0] - min_xy[0], 1.0), (screen_size[1] - 96) / max(max_xy[1] - min_xy[1], 1.0))
    l = max(length * scale, 12.0)
    w = max(width * scale, 8.0)
    c, s = math.cos(-yaw), math.sin(-yaw)
    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    rot = np.array([[c, -s], [s, c]])
    pts = corners @ rot.T + center
    pygame.draw.polygon(surface, (238, 183, 58), pts)
    pygame.draw.polygon(surface, (77, 65, 32), pts, width=2)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--rollout", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    rollout_path = resolve_path(args.rollout or config["viewer"]["rollout_path"], config, must_exist=True)
    data = np.load(rollout_path)
    frames = data["frames"]
    track = data["track"]
    left_boundary = data["left_boundary"] if "left_boundary" in data else np.empty((0, 2), dtype=np.float32)
    right_boundary = data["right_boundary"] if "right_boundary" in data else np.empty((0, 2), dtype=np.float32)
    lane_segments = data["lane_segments"] if "lane_segments" in data else np.empty((0, 2, 2), dtype=np.float32)
    lane_edges = lane_edge_polylines_from_config(config) or lane_edge_polylines(lane_segments)
    vehicle_length = float(data["vehicle_length"])
    vehicle_width = float(data["vehicle_width"])

    pygame.init()
    screen_size = (1100, 800)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Kart RL Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    bound_points = [track, frames[:, 1:3]]
    if len(left_boundary) and len(right_boundary):
        bound_points.extend([left_boundary, right_boundary])
    bound_points.extend(lane_edges)
    all_points = np.vstack(bound_points)
    bounds = (all_points.min(axis=0) - 4.0, all_points.max(axis=0) + 4.0)
    reference_line = world_to_screen(track, bounds, screen_size).astype(int)
    driven_line = world_to_screen(frames[:, 1:3], bounds, screen_size).astype(int)

    idx = 0
    paused = False
    running = True
    fps = int(config["viewer"].get("fps", 50))
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    idx = 0
                elif event.key == pygame.K_RIGHT:
                    idx = min(idx + fps, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    idx = max(idx - fps, 0)

        if not paused:
            idx = min(idx + 1, len(frames) - 1)

        screen.fill((236, 238, 232))
        for edge in lane_edges:
            pts = world_to_screen(edge, bounds, screen_size).astype(int)
            pygame.draw.lines(screen, (116, 120, 114), True, pts, width=1)
        pygame.draw.lines(screen, (34, 92, 132), True, reference_line, width=2)
        if idx > 1:
            pygame.draw.lines(screen, (198, 58, 42), False, driven_line[: idx + 1], width=2)

        frame = frames[idx]
        draw_vehicle(screen, frame[1:3], float(frame[3]), vehicle_length, vehicle_width, bounds, screen_size)
        text = f"t={frame[0]:6.2f}s  v={frame[4]:4.2f}m/s  progress={frame[7] * 100:5.1f}%  lateral={frame[8]:+.2f}m"
        screen.blit(font.render(text, True, (25, 29, 33)), (18, 16))
        screen.blit(font.render("space: pause  left/right: seek  r: restart", True, (72, 76, 80)), (18, 42))
        if idx == len(frames) - 1:
            screen.blit(font.render("rollout end", True, (170, 40, 40)), (18, 68))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()
