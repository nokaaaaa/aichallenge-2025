from typing import List, Tuple
import csv
import math

def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02}:{remaining_seconds:06.3f}"

def m_per_sec_to_kmh(m_per_sec: float) -> float:
    return m_per_sec * 3.6

def kmh_to_m_per_sec(kmh: float) -> float:
    return kmh / 3.6

def load_waypoints(csv_file_path: str) -> Tuple[List[float], List[float]]:
    with open(csv_file_path, newline="") as f:
        rows = list(csv.DictReader(f))
    wp_x = [float(row['wp_x']) for row in rows]
    wp_y = [float(row['wp_y']) for row in rows]
    return wp_x, wp_y

def load_ref_path(csv_file_path: str):
    with open(csv_file_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Reference path CSV is empty: {csv_file_path}")

    columns = set(rows[0].keys())
    if {'x_m', 'y_m', 'psi_rad', 'kappa_radpm'}.issubset(columns):
        x = [float(row['x_m']) for row in rows]
        y = [float(row['y_m']) for row in rows]
        psi = [float(row['psi_rad']) for row in rows]
        kappa = [float(row['kappa_radpm']) for row in rows]
        return x, y, psi, kappa

    if {'x', 'y', 'z_quat', 'w_quat'}.issubset(columns):
        x = [float(row['x']) for row in rows]
        y = [float(row['y']) for row in rows]
        psi = [
            math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)
            for z, w in (
                (float(row['z_quat']), float(row['w_quat'])) for row in rows
            )
        ]
        kappa = [0.0] * len(x)
        for i in range(1, len(x) - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            ds = math.hypot(dx, dy)
            dpsi = math.atan2(math.sin(psi[i + 1] - psi[i]), math.cos(psi[i + 1] - psi[i]))
            kappa[i] = dpsi / ds if ds > 1e-6 else 0.0
        if len(kappa) > 1:
            kappa[0] = kappa[1]
            kappa[-1] = kappa[-2]
        return x, y, psi, kappa

    raise ValueError(
        f"Unsupported reference path CSV columns in {csv_file_path}: {list(rows[0].keys())}")
