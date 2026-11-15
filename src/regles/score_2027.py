"""Score de la table officielle 2027 et barème associé."""

from math import cos, sin

from ..domaine.constantes import CONTACT_GAP
from ..domaine.geometrie import (
    _polygon_bounds, polygon_contact, rectangle_corners, throne_room_score as _throne_room_score,
    table_contact as _table_contact,
)
from ..domaine.modele import Block, Point, Rectangle
from ..domaine.score import construction_score as _construction_score
from ..terrains.officiel_2027 import (
    TABLE_HEIGHT, TABLE_WIDTH, THRONE_ROOM, TOWER_CENTERS, TOWER_RADIUS, WALL_ZONES,
)
from .bareme import Bareme2027


Bareme = Bareme2027
_BOUNDED_WALL_ZONES = tuple((zone, _polygon_bounds(zone)) for zone in WALL_ZONES)


def scoring_zones(block: Block) -> tuple[tuple[int, ...], bool]:
    left, top, right, bottom = _polygon_bounds(block.corners)
    # La tolérance sur les axes obliques peut dépasser CONTACT_GAP une fois projetée sur x ou y.
    gap = 2 * CONTACT_GAP
    walls = tuple(index for index, (zone, (x1, y1, x2, y2)) in enumerate(_BOUNDED_WALL_ZONES)
                  if x1 <= right + gap and left <= x2 + gap and y1 <= bottom + gap and top <= y2 + gap
                  and polygon_contact((0, 0), (0, 0), zone, block.corners, include_boundary=True) < 1)
    if walls:
        return walls, True
    origin_x, origin_y = block.position
    cosine, sine = cos(-block.angle), sin(-block.angle)
    corner_x, corner_y = block.corners[0][0] - origin_x, block.corners[0][1] - origin_y
    half_x = corner_x * cosine - corner_y * sine
    half_y = corner_x * sine + corner_y * cosine
    for center in TOWER_CENTERS:
        dx, dy = center[0] - origin_x, center[1] - origin_y
        x = dx * cosine - dy * sine
        y = dx * sine + dy * cosine
        gap_x = max(0, abs(x) - abs(half_x))
        gap_y = max(0, abs(y) - abs(half_y))
        if gap_x * gap_x + gap_y * gap_y <= (TOWER_RADIUS + CONTACT_GAP) ** 2:
            return (), True
    return (), False


def free_wall_slots(blocks) -> int:
    occupied = {zone for block in blocks for zone in scoring_zones(block)[0]}
    return len(WALL_ZONES) - len(occupied)


def construction_score(sites, bareme: Bareme2027 = Bareme2027()) -> int:
    return _construction_score(sites, bareme)


def table_contact(position: Point, delta: Point, footprint: tuple[Point, ...]) -> float:
    return _table_contact(position, delta, footprint, width=TABLE_WIDTH, height=TABLE_HEIGHT)


def throne_room_score(position: Point, footprint: tuple[Point, ...]) -> int:
    return _throne_room_score(position, footprint, THRONE_ROOM)
