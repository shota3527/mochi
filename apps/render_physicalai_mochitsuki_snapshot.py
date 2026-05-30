#!/usr/bin/env python3
"""Run a versioned PhysicalAI mochitsuki offline demo snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_DIR = REPO_ROOT / "demos" / "physicalai_mochitsuki" / "versions"
DEFAULT_UNITREE_G1_DIR = Path("/home/shota/dev/unitree/unitree_mujoco/unitree_robots/g1")


def default_unitree_g1_dir() -> Path:
    env_path = os.environ.get("UNITREE_G1_DIR")
    if env_path:
        return Path(env_path).expanduser()
    local_project_path = REPO_ROOT / "unitree_mujoco" / "unitree_robots" / "g1"
    if local_project_path.exists():
        return local_project_path
    return DEFAULT_UNITREE_G1_DIR


def load_snapshot_module(version: str):
    version_dir = VERSIONS_DIR / version
    module_path = version_dir / "mochitsuki_demo.py"
    if not module_path.exists():
        raise SystemExit(f"Unknown or incomplete snapshot version: {version}")

    spec = importlib.util.spec_from_file_location(f"physicalai_mochitsuki_{version}", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load snapshot module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=sorted(path.name for path in VERSIONS_DIR.iterdir() if path.is_dir()))
    parser.add_argument(
        "--unitree-g1-dir",
        type=Path,
        default=default_unitree_g1_dir(),
        help="Directory containing g1_29dof.xml and meshes.",
    )
    parser.add_argument(
        "demo_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected mochitsuki_demo.py snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version_dir = VERSIONS_DIR / args.version
    scene_template = version_dir / "scene_mochitsuki.xml"
    if not scene_template.exists():
        raise SystemExit(f"Missing scene template: {scene_template}")

    unitree_g1_dir = args.unitree_g1_dir.expanduser().resolve()
    if not (unitree_g1_dir / "g1_29dof.xml").exists():
        raise SystemExit(f"Missing g1_29dof.xml under --unitree-g1-dir={unitree_g1_dir}")

    module = load_snapshot_module(args.version)
    module.PROJECT_ROOT = REPO_ROOT
    module.SCENE_TEMPLATE_PATH = scene_template
    module.SCENE_PATH = unitree_g1_dir / f"scene_physicalai_mochitsuki_{args.version}.xml"
    module.DEFAULT_RENDER_DIR = REPO_ROOT / "outputs" / "physicalai_mochitsuki" / args.version

    demo_args = list(args.demo_args)
    if demo_args and demo_args[0] == "--":
        demo_args = demo_args[1:]
    if not demo_args:
        demo_args = ["--mode", "check"]

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(version_dir / "mochitsuki_demo.py"), *demo_args]
        module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
