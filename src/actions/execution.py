"""Exécution d'un trajet et état d'action."""

from math import ceil, pi, remainder

from pymunk import Vec2d

from ..domaine.constantes import CONTACT_GAP
from ..domaine.geometrie import _polygon_bounds as _bounds, polygon_contact, rectangle_corners, travel_time
from ..physique.constantes import TURN_SPEED
from ..regles.phases import FRAME_STEP
from ..regles.score_2027 import table_contact
from ..planification.base import Approach


def _follow(simulation: Simulation, approach: Approach, *, advance=None, deadline=None) -> bool:
    advance = simulation.advance if advance is None else advance
    if deadline is not None and simulation.time + approach.duration > deadline:
        return False
    for after_turn, targets in enumerate((approach.before, approach.after)):
        if after_turn and approach.turn and not approach.rolling:
            steps = ceil(abs(approach.turn) / TURN_SPEED / FRAME_STEP)
            step_time = abs(approach.turn) / TURN_SPEED / steps
            for _ in range(steps):
                if deadline is not None and simulation.time + step_time > deadline:
                    return False
                simulation.turn_by(approach.turn / steps)
                if simulation.moving:
                    simulation.move_to(*simulation.position)
                    return False
                if simulation.movement.blocked:
                    return False
                advance(step_time)
        for index, target in enumerate(targets):
            distance = Vec2d(*simulation.position).get_distance(target)
            if distance < 1e-7:
                continue
            simulation.move_to(*target)
            heading = simulation.heading
            if approach.rolling and not after_turn and index == len(targets) - 1:
                heading += approach.turn
                simulation.turn_by(approach.turn)
            remaining = travel_time(distance) + 1
            while simulation.moving and remaining > 0:
                step = FRAME_STEP if deadline is None else min(FRAME_STEP, deadline - simulation.time)
                if step <= 0:
                    break
                advance(step)
                remaining -= step
            if simulation.moving:
                simulation.move_to(*simulation.position)
                return False
            if Vec2d(*simulation.position).get_distance(target) > 2 * CONTACT_GAP:
                return False
            if abs(remainder(simulation.heading - heading, 2 * pi)) > 1e-7:
                return False
    return True


def _action_state(simulation):
    held = {simulation.held_pile(cup) for cup in range(1, len(simulation.geometry.mounts) + 1)}
    polygons = (*map(rectangle_corners, simulation.obstacles),
                *(block.corners for index, block in enumerate(simulation.blocks) if index not in held))
    return Vec2d(*simulation.position), tuple(simulation.footprints()), tuple((p, _bounds(p)) for p in polygons)


def _stroke_end(start, delta, footprints, obstacles):
    if not delta.length:
        return start
    fraction = 1.0
    for shape in footprints:
        fraction = min(fraction, table_contact(start, delta, shape))
        for corners, _ in obstacles:
            fraction = min(fraction, polygon_contact(start, delta, corners, shape))
    if fraction < 1:
        fraction = max(0, fraction - CONTACT_GAP / delta.length)
    return start + delta * fraction


