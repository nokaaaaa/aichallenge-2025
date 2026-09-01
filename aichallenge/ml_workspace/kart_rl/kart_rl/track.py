from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Projection:
    s: float
    x: float
    y: float
    yaw: float
    lateral_error: float
    heading_error: float
    curvature: float


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class Track:
    def __init__(
        self,
        points: np.ndarray,
        half_width_m: float,
        curvature_scale: float = 12.0,
        left_boundary: np.ndarray | None = None,
        right_boundary: np.ndarray | None = None,
    ):
        if len(points) < 3:
            raise ValueError("Track requires at least 3 points")
        self.points = points.astype(np.float64)
        self.left_boundary = left_boundary.astype(np.float64) if left_boundary is not None else None
        self.right_boundary = right_boundary.astype(np.float64) if right_boundary is not None else None
        self.half_width_m = float(half_width_m)
        self.curvature_scale = float(curvature_scale)

        self.seg_vecs = np.roll(self.points, -1, axis=0) - self.points
        self.seg_lengths = np.linalg.norm(self.seg_vecs, axis=1)
        keep = self.seg_lengths > 1e-6
        self.points = self.points[keep]
        self.seg_vecs = np.roll(self.points, -1, axis=0) - self.points
        self.seg_lengths = np.linalg.norm(self.seg_vecs, axis=1)
        self.seg_dirs = self.seg_vecs / self.seg_lengths[:, None]
        self.seg_yaws = np.arctan2(self.seg_dirs[:, 1], self.seg_dirs[:, 0])
        self.s_starts = np.concatenate([[0.0], np.cumsum(self.seg_lengths[:-1])])
        self.length = float(np.sum(self.seg_lengths))
        self.curvatures = self._compute_curvatures()

    @classmethod
    def from_csv(cls, path, half_width_m: float, curvature_scale: float = 12.0) -> "Track":
        data = np.genfromtxt(path, delimiter=",", names=True)
        points = np.column_stack([data["x"], data["y"]])
        points = points - points[0]
        return cls(points=points, half_width_m=half_width_m, curvature_scale=curvature_scale)

    @classmethod
    def from_lane_csv(cls, path: str | Path, half_width_m: float | str, curvature_scale: float = 12.0) -> "Track":
        data = np.genfromtxt(path, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError("lane CSV must have at least 4 columns: left_x,left_y,right_x,right_y")
        left = _clean_polyline(data[:, 0:2])
        right = _clean_polyline(data[:, 2:4])
        sample_count = max(len(left), len(right))
        left = _resample_closed_polyline(left, sample_count)
        right = _resample_closed_polyline(right, sample_count)
        widths = np.linalg.norm(right - left, axis=1)
        center = 0.5 * (left + right)
        origin = center[0].copy()
        center = center - origin
        if isinstance(half_width_m, str):
            if half_width_m != "auto":
                raise ValueError("track.half_width_m must be a number or 'auto'")
            half_width = float(np.nanmedian(widths) * 0.5)
        else:
            half_width = float(half_width_m)
        return cls(
            points=center,
            half_width_m=half_width,
            curvature_scale=curvature_scale,
            left_boundary=left - origin,
            right_boundary=right - origin,
        )

    def _compute_curvatures(self) -> np.ndarray:
        yaw_delta = wrap_angle(np.roll(self.seg_yaws, -1) - self.seg_yaws)
        avg_len = 0.5 * (self.seg_lengths + np.roll(self.seg_lengths, -1))
        return yaw_delta / np.maximum(avg_len, 1e-6)

    def project(self, x: float, y: float, yaw: float) -> Projection:
        p = np.array([x, y], dtype=np.float64)
        rel = p - self.points
        t = np.clip(np.einsum("ij,ij->i", rel, self.seg_dirs) / self.seg_lengths, 0.0, 1.0)
        closest = self.points + self.seg_vecs * t[:, None]
        dist2 = np.sum((p - closest) ** 2, axis=1)
        idx = int(np.argmin(dist2))
        center = closest[idx]
        seg_dir = self.seg_dirs[idx]
        delta = p - center
        lateral = float(seg_dir[0] * delta[1] - seg_dir[1] * delta[0])
        track_yaw = float(self.seg_yaws[idx])
        return Projection(
            s=float((self.s_starts[idx] + t[idx] * self.seg_lengths[idx]) % self.length),
            x=float(center[0]),
            y=float(center[1]),
            yaw=track_yaw,
            lateral_error=lateral,
            heading_error=float(wrap_angle(yaw - track_yaw)),
            curvature=float(self.curvatures[idx]),
        )

    def sample_at(self, s: float) -> tuple[float, float, float, float]:
        s = s % self.length
        idx = int(np.searchsorted(self.s_starts + self.seg_lengths, s, side="right"))
        idx = min(idx, len(self.points) - 1)
        ratio = (s - self.s_starts[idx]) / self.seg_lengths[idx]
        point = self.points[idx] + self.seg_vecs[idx] * ratio
        return float(point[0]), float(point[1]), float(self.seg_yaws[idx]), float(self.curvatures[idx])

    def lookahead_curvatures(self, s: float, distances: tuple[float, ...]) -> np.ndarray:
        return np.array([self.sample_at(s + d)[3] * self.curvature_scale for d in distances], dtype=np.float32)


def _clean_polyline(points: np.ndarray) -> np.ndarray:
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError("lane boundary must contain at least 3 finite points")
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], step > 1e-6])
    return points[keep]


def _resample_closed_polyline(points: np.ndarray, sample_count: int) -> np.ndarray:
    closed = np.vstack([points, points[0]])
    seg = np.diff(closed, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-6
    starts = closed[:-1][keep]
    vecs = seg[keep]
    lens = seg_len[keep]
    cumulative = np.concatenate([[0.0], np.cumsum(lens)])
    total = cumulative[-1]
    samples = np.linspace(0.0, total, sample_count, endpoint=False)
    indices = np.searchsorted(cumulative[1:], samples, side="right")
    ratio = (samples - cumulative[indices]) / lens[indices]
    return starts[indices] + vecs[indices] * ratio[:, None]
