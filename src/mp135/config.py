"""Read project settings relative to the repository, not the shell directory."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    with open(path or ROOT / "params.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(config, name):
    return ROOT / config["paths"][name]
