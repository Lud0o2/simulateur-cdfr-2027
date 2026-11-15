"""Scénarios d'évaluation."""

from ..domaine.scenario import ScenarioSpec
from ..robots.loader import load_robot_spec
from ..terrains.officiel_2027 import TERRAIN


def officiel_2027() -> ScenarioSpec:
    return ScenarioSpec(TERRAIN, load_robot_spec(), TERRAIN.obstacles, TERRAIN.initial_blocks)
