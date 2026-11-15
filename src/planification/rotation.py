"""Rotation en roulant et pivot sur place."""

from functools import lru_cache
from math import atan2, ceil, floor, hypot, pi

from ..domaine.geometrie import _polygon_bounds as _bounds, rectangle_corners
from ..physique.moteur import Simulation


def _turn_envelope(footprint, rotated, angle):
    """Rectangle qui couvre la trajectoire des sommets pendant le changement de cap."""
    points = [*footprint, *rotated]
    for x, y in footprint:
        heading = atan2(y, x)
        start, end = sorted((heading, heading + angle))
        radius = hypot(x, y)
        extremes = ((radius, 0), (0, radius), (-radius, 0), (0, -radius))
        # Les extrêmes d'un arc se trouvent à ses extrémités ou sur les axes.
        for axis in range(ceil(start / (pi / 2)), floor(end / (pi / 2)) + 1):
            points.append(extremes[axis % 4])
    return rectangle_corners(_bounds(tuple(points)))


# Plusieurs poses de dépose peuvent demander le même pivot.
@lru_cache(maxsize=16384)
def _pivot_clear(position, footprints, angle, polygons):
    return Simulation.rotation_clear(position, footprints, angle, polygons)


