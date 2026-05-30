#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"
uv run --project "${PROJECT_ROOT}" python demos/mochitsuki/mochitsuki_demo.py --mode check --render-smoke
exec uv run --project "${PROJECT_ROOT}" python demos/mochitsuki/mochitsuki_demo.py \
  --mode render \
  --camera-layout multi \
  --save-keyframes \
  --duration 6.2 \
  --fps 30 \
  --width 1920 \
  --height 1080 \
  --output demos/mochitsuki/renders/g1_mochitsuki_multi_angle_demo.mp4
