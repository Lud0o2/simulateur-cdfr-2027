"""Modèle de données pur du simulateur."""

from dataclasses import dataclass
from typing import Any


Point = tuple[float, float]
Rectangle = tuple[float, float, float, float]
Pose = tuple[float, float, float]  # x, y, cap en radians


@dataclass(frozen=True)
class Movement:
    end: Point
    blocked: bool
    turn: bool = False
    rotation_blocked: bool = False


@dataclass(frozen=True)
class Block:
    position: Point
    angle: float
    corners: tuple[Point, ...]


@dataclass(frozen=True)
class SuctionCup:
    position: Point
    enabled: bool


@dataclass(frozen=True)
class EtatVentouse:
    position: Point
    enabled: bool
    construction: str = "mur"


@dataclass(frozen=True)
class Candidat:
    action: str
    gain: float
    duration_s: float
    robot: str = "robot"
    payload: Any = None


@dataclass(frozen=True)
class Observation:
    robot: Pose
    piles: tuple[Block, ...]
    tenues: dict[int, int]
    score: int
    temps: float
