"""Catalogue des poses de dépôt."""

from math import radians

from pymunk import Vec2d

from ..terrains.officiel_2027 import WALL_ZONES


def _drop_sites():
    for a, b, c, d in (tuple(Vec2d(*corner) for corner in zone) for zone in WALL_ZONES):
        center = (a + c) / 2
        for length in (0, -0.4, 0.4):
            for width in (0, -0.6, 0.6):
                yield center + (b - a) * length + (d - a) * width


DROP_SITES = tuple(_drop_sites())
CENTERED_DROPS = tuple(
    ((a + c) / 2, (b - a).angle)
    for a, b, c, _ in (tuple(Vec2d(*corner) for corner in zone) for zone in WALL_ZONES)
)
ALIGNED_ANGLES = (0,) * len(CENTERED_DROPS)
# Les deux sens d'approche couvrent aussi +90° pour la même empreinte qu'à −90°.
DROP_ANGLES = tuple(radians(angle) for angle in range(-90, 90, 10))
ALTERNATE_DROPS = tuple(
    (site + Vec2d(length, width).rotated(axis), axis + rotation)
    for site, axis in CENTERED_DROPS
    for length, width in ((0, 0), (0, -120), (0, 120), (-120, -160))
    for rotation in DROP_ANGLES
)


def _shifted_drops(offsets, angles):
    return tuple((site + Vec2d(*offset).rotated(axis), axis + rotation)
                 for (site, axis), offset, rotation in zip(CENTERED_DROPS, offsets, angles, strict=True))
