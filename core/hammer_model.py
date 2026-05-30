"""Simplified primitive hammer model helpers."""

from __future__ import annotations

from dataclasses import dataclass


GRAVITY = 9.81


@dataclass(frozen=True)
class HammerModel:
    total_mass_kg: float
    head_mass_kg: float
    handle_length_m: float
    wrist_to_com_m: float
    com_from_adapter_m: tuple[float, float, float]

    @property
    def static_wrist_moment_nm(self) -> float:
        return self.total_mass_kg * GRAVITY * self.wrist_to_com_m

    @property
    def handle_mass_kg(self) -> float:
        return self.total_mass_kg - self.head_mass_kg


def load_hammer_model(config: dict) -> HammerModel:
    com = config.get("center_of_mass_from_wrist_m")
    if com is None:
        com = config["center_of_mass_from_adapter_m"]
    return HammerModel(
        total_mass_kg=float(config["hammer_total_mass_kg"]),
        head_mass_kg=float(config["hammer_head_mass_kg"]),
        handle_length_m=float(config["handle_length_m"]),
        wrist_to_com_m=float(config["wrist_to_com_distance_m"]),
        com_from_adapter_m=tuple(float(v) for v in com),
    )
