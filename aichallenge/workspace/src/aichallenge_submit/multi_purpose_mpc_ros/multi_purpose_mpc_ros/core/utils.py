from typing import List, Tuple
import pandas as pd

def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02}:{remaining_seconds:06.3f}"

def m_per_sec_to_kmh(m_per_sec: float) -> float:
    return m_per_sec * 3.6

def kmh_to_m_per_sec(kmh: float) -> float:
    return kmh / 3.6

def load_waypoints(csv_file_path: str) -> Tuple[List[float], List[float]]:
    df = pd.read_csv(csv_file_path)
    wp_x = df['wp_x'].tolist()
    wp_y = df['wp_y'].tolist()
    return wp_x, wp_y

def load_ref_path(csv_file_path: str):
    df = pd.read_csv(csv_file_path)
    if {'x_m', 'y_m', 'psi_rad', 'kappa_radpm'}.issubset(df.columns):
        x = df['x_m'].tolist()
        y = df['y_m'].tolist()
        psi = df['psi_rad'].tolist()
        kappa = df['kappa_radpm'].tolist()
    elif {'x', 'y'}.issubset(df.columns):
        # simple_trajectory_generator's traj.csv. ReferencePath recomputes
        # heading and curvature from the XY centerline.
        x = df['x'].tolist()
        y = df['y'].tolist()
        psi = [0.0] * len(x)
        kappa = [0.0] * len(x)
    else:
        raise ValueError(
            f"Unsupported reference path CSV columns in {csv_file_path}: "
            f"{list(df.columns)}")
    return x, y, psi, kappa
