from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]


def load_config(path: str | Path | None) -> dict[str, Any]:
    config_path = Path(path) if path else PACKAGE_ROOT / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_path(path: str | Path, config: dict[str, Any], must_exist: bool = False) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    else:
        config_dir = Path(config["_config_path"]).resolve().parent
        from_config = (config_dir / candidate).resolve()
        from_package = (PACKAGE_ROOT / candidate).resolve()
        from_repo = (REPO_ROOT / candidate).resolve()
        if must_exist:
            if from_config.exists():
                resolved = from_config
            elif from_package.exists():
                resolved = from_package
            else:
                resolved = from_repo
        elif from_config.exists():
            resolved = from_config
        else:
            resolved = from_package
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved
