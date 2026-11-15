"""Génération des approches de prise et de dépose."""

from dataclasses import replace
from math import copysign, pi, remainder

from pymunk import Vec2d

from ..domaine.constantes import CONTACT_GAP
from ..domaine.geometrie import travel_time
from ..domaine.modele import Block
from ..planification.astar import _approach, _clear
from ..planification.base import DemandeTrajet, Pickup, approach_depuis_trajet
from ..regles.score_2027 import scoring_zones
from ..terrains.officiel_2027 import TOWER_CENTERS, TOWER_RADIUS
from .execution import _stroke_end
from .params import PICKUP_GAP, PICKUP_STANDOFF, PICKUP_STROKE


def _planifier(planner, start, end, angle, footprints, obstacles):
    if planner is None:
        return _approach(start, end, angle, footprints, obstacles)
    demande = DemandeTrajet(
        tuple(start), tuple(end),
        tuple(tuple(tuple(point) for point in footprint) for footprint in footprints),
        tuple(obstacle for obstacle, _ in obstacles),
        angle_vise=angle,
    )
    trajet = planner.planifier(demande)
    return None if trajet is None else approach_depuis_trajet(trajet)


def _pickup_approach(simulation, block, cup, state, planner=None):
    mount, facing = simulation.geometry.mounts[cup - 1]
    start, footprints, obstacles = state
    center = Vec2d(*block.position)
    tangent = facing.perpendicular()
    lateral_gaps = [(Vec2d(*other) - mount).dot(tangent)
                    for other, normal in simulation.geometry.mounts
                    if normal == facing and other != mount
                    and abs((Vec2d(*other) - mount).dot(facing)) <= CONTACT_GAP]
    offset = 0
    if len(lateral_gaps) == 1:
        gap = lateral_gaps[0]
        half_length = max((Vec2d(*corner) - center).rotated(-block.angle).x for corner in block.corners)
        # Garde 10 mm entre deux piles identiques prises sur la même face.
        offset = copysign(max(0, half_length + 5 - abs(gap) / 2), -gap)
    pickup = Vec2d(*mount) + tangent * offset
    choices = []
    for side in (-1, 1):
        normal = Vec2d(0, side).rotated(block.angle)
        half_width = max((Vec2d(*corner) - center).dot(normal) for corner in block.corners)
        heading = (-normal).angle - facing.angle
        angle = remainder(heading - simulation.heading, 2 * pi)
        if abs(angle) < 1e-12:
            angle = 0
        contact = center + normal * (half_width + PICKUP_GAP) - pickup.rotated(heading)
        standoff = contact + normal * (PICKUP_STANDOFF - PICKUP_GAP)
        rotated = tuple(tuple(Vec2d(*corner).rotated(angle) for corner in shape) for shape in footprints)
        if not _clear(standoff, contact, rotated, obstacles):
            continue
        approach = _planifier(planner, start, standoff, angle, footprints, obstacles)
        if approach is not None:
            approach = replace(approach, duration=approach.duration + travel_time(standoff.get_distance(contact)),
                               after=(*approach.after, contact))
            end = standoff - normal * PICKUP_STROKE
            held = tuple(Vec2d(*corner) - contact for corner in block.corners)
            others = tuple(obstacle for obstacle in obstacles if obstacle[0] != block.corners)
            choices.append(Pickup(approach, _stroke_end(contact, end - contact, (*rotated, held), others)))
    return min(choices, key=lambda option: option.duration, default=None)


def _wall_drop_zones(block: Block):
    center = Vec2d(*block.position)
    if any(center.get_distance(tower) <= TOWER_RADIUS + CONTACT_GAP for tower in TOWER_CENTERS):
        return (), False
    walls, _ = scoring_zones(block)
    return walls, bool(walls)


def _drop_target(block, destination, angle, start):
    center = Vec2d(*block.position)
    turn = remainder(remainder(angle, 2 * pi) - block.angle, 2 * pi)
    return destination - (center - start).rotated(turn), turn


def _drop_approach(block, destination, angle, state, planner=None):
    start, footprints, obstacles = state
    end, turn = _drop_target(block, destination, angle, start)
    center = Vec2d(*block.position)
    corners = tuple(tuple(destination + (Vec2d(*corner) - center).rotated(turn)) for corner in block.corners)
    if not _wall_drop_zones(Block(tuple(destination), block.angle + turn, corners))[1]:
        return None
    return _planifier(planner, start, end, turn, footprints, obstacles)


