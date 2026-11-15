"""Dépose d'une pile à une pose cible."""

from math import pi, remainder

from pymunk import Vec2d

from ..domaine.geometrie import travel_time
from ..domaine.validation import _finite_number
from ..physique.moteur import Simulation
from ..planification.base import Approach
from ..terrains.officiel_2027 import TABLE_HEIGHT, TABLE_WIDTH
from .approche import _drop_approach, _wall_drop_zones
from .execution import _action_state, _follow, _stroke_end
from .params import DROP_RETREAT


def _drop_retreat(simulation, cup, *, advance=None, deadline=None):
    start, footprints, obstacles = _action_state(simulation)
    _, facing = simulation.geometry.mounts[cup - 1]
    end = _stroke_end(start, -DROP_RETREAT * facing.rotated(simulation.heading), footprints, obstacles)
    _follow(simulation, Approach(travel_time(start.get_distance(end)), (), 0, (end,)),
            advance=advance, deadline=deadline)


def place_pile(simulation: Simulation, cup: int, x: float, y: float, angle: float, *, advance=None,
               deadline=None, on_release=None, planner=None) -> bool:
    """Dépose la pile à la pose demandée, puis recule si l'espace le permet."""
    if (simulation.moving or type(cup) is not int or not 1 <= cup <= len(simulation.geometry.mounts)
            or not all(_finite_number(value) for value in (x, y, angle))
            or not (0 <= x <= TABLE_WIDTH and 0 <= y <= TABLE_HEIGHT)):
        return False
    index = simulation.held_pile(cup)
    if index is None:
        return False
    block = simulation.blocks[index]
    destination = Vec2d(x, y)
    target_angle = remainder(angle, 2 * pi)
    approach = _drop_approach(block, destination, angle, _action_state(simulation), planner=planner)
    if approach is None or not _follow(simulation, approach, advance=advance, deadline=deadline):
        return False
    placed = simulation.blocks[index]
    if (Vec2d(*placed.position).get_distance(destination) > 1e-7
            or abs(remainder(placed.angle - target_angle, 2 * pi)) > 1e-7 or not _wall_drop_zones(placed)[1]):
        return False
    simulation.toggle_suction(cup)
    if on_release is not None:
        on_release()
    _drop_retreat(simulation, cup, advance=advance, deadline=deadline)
    return True




class Depose:
    id = "depose"
    construction = None

    def candidats(self, ctx):
        return ()

    def executer(self, ctx, candidat) -> bool:
        payload = candidat.payload
        return place_pile(
            ctx.moteur, payload["cup"], payload["x"], payload["y"], payload["angle"],
            deadline=ctx.deadline, planner=ctx.planificateur,
        )
