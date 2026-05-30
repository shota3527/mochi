#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"
uv run --project "${PROJECT_ROOT}" python demos/mochitsuki/mochitsuki_demo_v3.py --mode check --render-smoke
exec uv run --project "${PROJECT_ROOT}" python demos/mochitsuki/mochitsuki_demo_v3.py \
  --mode render \
  --camera-layout multi \
  --save-keyframes \
  --duration 5.2 \
  --fps 30 \
  --width 1920 \
  --height 1080 \
  --output demos/mochitsuki/renders/v3_g1_mochitsuki_multi_angle_demo.mp4
