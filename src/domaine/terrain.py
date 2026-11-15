"""Description d'un terrain de jeu."""

from dataclasses import dataclass

from .modele import Point, Pose, Rectangle


@dataclass(frozen=True)
class TerrainSpec:
    id: str
    width: float
    height: float
    start: Pose
    throne_room: Rectangle
    wall_zones: tuple[tuple[Point, ...], ...]
    tower_centers: tuple[Point, ...]
    tower_radius: float
    initial_blocks: tuple[Rectangle, ...]
    obstacles: tuple[Rectangle, ...] = ()
