#!/usr/bin/env python3
"""Read one G1 low-state sample and log joint order data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backends.sdk2_python_backend import G1Sdk2Backend


def load_joint_config() -> dict:
    with (PROJECT_ROOT / "configs" / "g1_29dof_joints.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    joint_config = load_joint_config()
    joints = joint_config["joints"]

    backend = G1Sdk2Backend(domain_id=args.domain_id, interface=args.interface)
    backend.initialize(enable_commands=False)
    try:
        sample = backend.wait_for_state(timeout_s=args.timeout)
    except TimeoutError as exc:
        raise SystemExit(
            f"{exc}\nStart the G1 simulator/DDS publisher first, then re-run this app."
        ) from exc

    print("G1 lowstate sample")
    print(f"domain_id={args.domain_id} interface={args.interface} mode_machine={sample.mode_machine}")
    print("idx  joint_name                 q(rad)       dq(rad/s)     tau_est(Nm)")
    for joint, q, dq, tau in zip(joints, sample.q, sample.dq, sample.tau_est):
        print(f"{joint['index']:>2}   {joint['name']:<24} {q:> .7f}  {dq:> .7f}  {tau:> .7f}")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"dump_state_{stamp}.json"
    payload = {
        "stamp": sample.stamp,
        "domain_id": args.domain_id,
        "interface": args.interface,
        "mode_machine": sample.mode_machine,
        "source": "rt/lowstate",
        "joint_order": [joint["name"] for joint in joints],
        "q": sample.q.tolist(),
        "dq": sample.dq.tolist(),
        "tau_est": sample.tau_est.tolist(),
    }
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"saved {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
