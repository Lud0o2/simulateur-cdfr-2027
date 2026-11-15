"""Robot rectangulaire de démonstration."""

from pathlib import Path

from ..loader import load_robot_spec


TOML = Path(__file__).with_name("robot.toml")


def construire():
    return load_robot_spec(TOML)
