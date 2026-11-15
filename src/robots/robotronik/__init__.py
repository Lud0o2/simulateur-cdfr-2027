"""Robot Robotronik : le TOML reste à la racine pendant la transition."""

from ..loader import ROBOT_FILE, load_robot_spec


TOML = ROBOT_FILE


def construire():
    return load_robot_spec(TOML)
