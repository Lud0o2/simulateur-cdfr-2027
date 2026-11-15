"""Planificateur direct : ligne droite si elle est libre."""

from ..domaine.geometrie import travel_time
from .astar import _clear
from .base import Approach


def planifier_direct(start, end, footprints, obstacles) -> Approach | None:
    if not _clear(start, end, footprints, obstacles):
        return None
    return Approach(travel_time(start.get_distance(end)), (), 0, (end,))
