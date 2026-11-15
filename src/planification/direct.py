"""Planificateur direct : ligne droite si elle est libre."""

from ..domaine.geometrie import travel_time
from .astar import _clear
from .base import Approach


def planifier_direct(start, end, footprints, obstacles) -> Approach | None:
    if not _clear(start, end, footprints, obstacles):
        return None
    return Approach(travel_time(start.get_distance(end)), (), 0, (end,))


class DirectPlanner:
    id = "direct"

    def planifier(self, demande):
        from pymunk import Vec2d

        from ..domaine.geometrie import _polygon_bounds
        from .base import Trajet

        obstacles = tuple((obstacle, _polygon_bounds(obstacle)) for obstacle in demande.obstacles)
        approach = planifier_direct(
            Vec2d(*demande.depart), Vec2d(*demande.arrivee), demande.empreintes, obstacles,
        )
        if approach is None:
            return None
        return Trajet(approach.duration, approach.before, approach.turn, approach.after, approach.rolling)
