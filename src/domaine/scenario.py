"""Scénario complet : terrain, robot, piles et obstacles."""

from dataclasses import dataclass, field

from .modele import Rectangle
from .robot import RobotSpec


@dataclass(frozen=True)
class ScenarioSpec:
    terrain: object
    robot: RobotSpec
    obstacles: tuple[Rectangle, ...] = ()
    piles_initiales: tuple[Rectangle, ...] = field(default_factory=tuple)
    graine: int | None = None
