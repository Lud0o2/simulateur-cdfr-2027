"""Prise d'une pile avec une ventouse."""

from pymunk import Vec2d

from ..domaine.geometrie import travel_time
from ..physique.moteur import Simulation
from ..planification.base import Approach
from .approche import _pickup_approach
from .execution import _action_state, _follow, _stroke_end


def take_pile(simulation: Simulation, pile: int, cup: int, *, advance=None, deadline=None) -> bool:
    """Rejoint un grand côté de la pile désignée et la saisit avec la ventouse choisie."""
    blocks = simulation.blocks
    if (simulation.moving or type(pile) is not int or not 0 <= pile < len(blocks)
            or type(cup) is not int or not 1 <= cup <= len(simulation.geometry.mounts)
            or simulation.held_pile(cup) is not None
            or any(simulation.held_pile(number) == pile for number in range(1, len(simulation.geometry.mounts) + 1))):
        return False
    pickup = _pickup_approach(simulation, blocks[pile], cup, _action_state(simulation))
    return pickup is not None and _take_pickup(simulation, pile, cup, pickup, advance=advance, deadline=deadline)


def _take_pickup(simulation, pile, cup, pickup, *, advance=None, deadline=None):
    if not _follow(simulation, pickup.approach, advance=advance, deadline=deadline):
        return False
    if not simulation.grip_pile(cup, pile):
        return False
    start, footprints, obstacles = _action_state(simulation)
    end = _stroke_end(start, pickup.end - start, footprints, obstacles)
    # La pile est déjà saisie, même si le temps restant ou un obstacle abrège la course.
    stroke = Approach(travel_time(end.get_distance(start)), (), 0, (end,))
    _follow(simulation, stroke, advance=advance, deadline=deadline)
    return True


