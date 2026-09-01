from __future__ import annotations

import argparse
import math
from pathlib import Path

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
    half_width = float(data["half_width"])
    vehicle_length = float(data["vehicle_length"])
    vehicle_width = float(data["vehicle_width"])

    pygame.init()
    screen_size = (1100, 800)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Kart RL Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    bounds = (track.min(axis=0) - half_width - 4.0, track.max(axis=0) + half_width + 4.0)
    center_line = world_to_screen(track, bounds, screen_size).astype(int)
    left_right = []
    closed = np.vstack([track, track[0]])
    dirs = np.diff(closed, axis=0)
    yaws = np.arctan2(dirs[:, 1], dirs[:, 0])
    normals = np.column_stack([-np.sin(yaws), np.cos(yaws)])
    left = track + normals * half_width
    right = track - normals * half_width
    left_right.append(world_to_screen(left, bounds, screen_size).astype(int))
    left_right.append(world_to_screen(right, bounds, screen_size).astype(int))

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
        for border in left_right:
            pygame.draw.lines(screen, (84, 88, 82), True, border, width=2)
        pygame.draw.lines(screen, (34, 92, 132), True, center_line, width=3)

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
