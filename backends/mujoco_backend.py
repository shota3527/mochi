"""MuJoCo backend placeholder for direct simulation access."""

from __future__ import annotations


class MujocoBackend:
    """Reserved for direct MuJoCo stepping after DDS state reading is confirmed."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def initialize(self) -> None:
        raise NotImplementedError("Use the Unitree DDS simulator path first.")
