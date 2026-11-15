"""Données du terrain officiel Eurobot 2027."""

from math import radians

import pymunk

from ..domaine.geometrie import rectangle_corners
from ..domaine.terrain import TerrainSpec


TABLE_WIDTH = 3000
TABLE_HEIGHT = 2000
START = (250.0, 1000.0)
THRONE_ROOM = (0, 750, 500, 1250)  # Règlement Eurobot 2027, § D.3 et plan G.1.
TABLE = ((0, 0), (TABLE_WIDTH, 0), (TABLE_WIDTH, TABLE_HEIGHT), (0, TABLE_HEIGHT))
# Règlement Eurobot 2027, § D.4 et plan G.1 (X depuis le côté jaune).
START_BLOCKS = (
    *((x - 160, y, x + 160, y + 110) for y in (0, 1890) for x in (750, 1250, 1750, 2250)),
    (1445, 340, 1555, 660),
    (1445, 1340, 1555, 1660),
)


def _wall_zone(x: float, y: float, angle: float = 0) -> tuple[tuple[float, float], ...]:
    return tuple(tuple(pymunk.Vec2d(x, y) + pymunk.Vec2d(*vertex).rotated(radians(angle)))
                 for vertex in rectangle_corners((-200, -100, 200, 100)))


# Remparts jaunes, § D.4 et G.1. Origine au coin supérieur gauche du tapis.
WALL_ZONES = (_wall_zone(250, 400), _wall_zone(675, 575, 45), _wall_zone(850, 1000, 90),
              _wall_zone(675, 1425, -45), _wall_zone(250, 1600))
TOWER_CENTERS = ((500, 400), (850, 750), (850, 1250), (500, 1600))
TOWER_RADIUS = 100


TERRAIN = TerrainSpec(
    id="officiel_2027",
    width=TABLE_WIDTH,
    height=TABLE_HEIGHT,
    start=(*START, 0.0),
    throne_room=THRONE_ROOM,
    wall_zones=WALL_ZONES,
    tower_centers=TOWER_CENTERS,
    tower_radius=TOWER_RADIUS,
    initial_blocks=START_BLOCKS,
)
