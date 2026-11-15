"""Stratégie de référence gloutonne."""

from math import ceil, pi, remainder

from pymunk import Vec2d

from ..actions.approche import _wall_drop_zones
from ..actions.depose import _drop_retreat
from ..actions.execution import _stroke_end
from ..actions.params import DROP_RETREAT, PICKUP_GAP, PICKUP_STANDOFF, PICKUP_STROKE
from ..actions.prise import _take_pickup
from ..domaine.geometrie import _polygon_bounds as _bounds, rectangle_corners, travel_time
from ..domaine.modele import Block
from ..physique.constantes import TURN_SPEED
from ..planification.astar import OPEN_SPACE, _clear, _route
from ..planification.base import Approach, Pickup, Plan
from ..regles.phases import BUILD_DEADLINE, FRAME_STEP
from ..regles.score_2027 import construction_score, scoring_zones
from .poses import DROP_SITES


def choose_plan(simulation: Simulation, unavailable: set[int]) -> Plan | None:
    """Classe les prises suivies d'un dépôt par gain de score par seconde."""
    blocks = simulation.blocks
    sites = tuple(map(scoring_zones, blocks))
    score = construction_score(sites)
    start = Vec2d(*simulation.position)
    fixed = tuple((corners, _bounds(corners)) for corners in map(rectangle_corners, simulation.obstacles))
    piles = tuple((block.corners, _bounds(block.corners)) for block in blocks)
    turns = {}
    candidates = []
    for index, block in enumerate(blocks):
        if index in unavailable or sites[index][1]:
            continue
        center = Vec2d(*block.position)
        relative = tuple(Vec2d(*corner) - center for corner in block.corners)
        obstacles = (*fixed, *piles[:index], *piles[index + 1:])
        deposits = []
        for site in DROP_SITES:
            corners = tuple(tuple(site + corner) for corner in relative)
            zones = _wall_drop_zones(Block(tuple(site), block.angle, corners))
            if zones[1] and _clear(site, site, (relative,), obstacles):
                gain = construction_score((*sites, zones)) - score
                deposits.append((gain, site, (corners, _bounds(corners))))

        if not deposits:
            continue
        pickup_obstacles = (*obstacles, piles[index])
        for side in (-1, 1):
            normal = Vec2d(0, side).rotated(block.angle)
            half_width = max(corner.dot(normal) for corner in relative)
            for cup, (mount, facing) in enumerate(simulation.geometry.mounts, 1):
                heading = (-normal).angle - facing.angle
                angle = remainder(heading - simulation.heading, 2 * pi)
                if angle not in turns:
                    turns[angle] = simulation.can_turn_freely(angle)
                if not turns[angle]:
                    continue
                footprint = tuple(Vec2d(*corner).rotated(heading) for corner in simulation.geometry.contour)
                contact = center + normal * (half_width + PICKUP_GAP) - Vec2d(*mount).rotated(heading)
                standoff = contact + normal * (PICKUP_STANDOFF - PICKUP_GAP)
                if not _clear(standoff, contact, (footprint,), pickup_obstacles):
                    continue
                approach, approach_time = _route(start, standoff, (footprint,), pickup_obstacles)
                if not approach:
                    continue
                held = tuple(corner + center - contact for corner in relative)
                end = _stroke_end(contact, standoff - normal * PICKUP_STROKE - contact, (footprint, held), obstacles)
                approach = (*approach, contact)
                turn_frames = ceil(abs(angle) / TURN_SPEED / FRAME_STEP)
                approach_time += (turn_frames + len(approach)) * FRAME_STEP + travel_time(PICKUP_STANDOFF - PICKUP_GAP)
                pickup = Pickup(Approach(approach_time, (), angle, approach), end)
                for gain, site, placed in deposits:
                    drop = site + contact - center
                    duration = (pickup.duration + 4 * FRAME_STEP
                                + travel_time(drop.get_distance(end)) + travel_time(drop.get_distance(OPEN_SPACE)))
                    if simulation.time + duration <= BUILD_DEADLINE:
                        candidates.append((gain / duration, gain, duration, index, cup, pickup,
                                           drop, footprint, held, obstacles, placed))

    best, best_rate = None, 0
    for rate, gain, duration, index, cup, pickup, drop, footprint, held, obstacles, placed in (
        sorted(candidates, key=lambda item: -item[0])
    ):
        if rate <= best_rate:
            break
        if not _clear(drop, drop, (footprint, held), obstacles):
            continue
        drop_obstacles = (*obstacles, placed)
        _, facing = simulation.geometry.mounts[cup - 1]
        backward = -DROP_RETREAT * facing.rotated(simulation.heading + pickup.approach.turn)
        retreat_start = _stroke_end(drop, backward, (footprint,), drop_obstacles)
        retreat, back_time = _route(retreat_start, OPEN_SPACE, (footprint,), drop_obstacles)
        if not retreat:
            continue
        transport, transport_time = _route(pickup.end, drop, (footprint, held), obstacles)
        if not transport:
            continue
        duration += (transport_time - travel_time(pickup.end.get_distance(drop))
                     + travel_time(drop.get_distance(retreat_start)) + back_time - travel_time(drop.get_distance(OPEN_SPACE)))
        duration += (len(transport) + len(retreat) - 2) * FRAME_STEP
        if simulation.time + duration <= BUILD_DEADLINE and gain / duration > best_rate:
            best = Plan(index, cup, pickup, transport, retreat)
            best_rate = gain / duration
    return best


def _play_reference(simulation, record):
    def advance(label):
        record(label)
        simulation.advance(min(FRAME_STEP, BUILD_DEADLINE - simulation.time))

    def move(target, label):
        if simulation.time >= BUILD_DEADLINE:
            return False
        simulation.move_to(*target)
        while simulation.moving and simulation.time < BUILD_DEADLINE:
            advance(label)
        return not simulation.moving and Vec2d(*simulation.position).get_distance(target) < 0.01

    unavailable = set()
    while simulation.time < BUILD_DEADLINE:
        plan = choose_plan(simulation, unavailable)
        if plan is None:
            break
        name = f"pile {plan.pile + 1}"
        def approach_step(seconds):
            simulation.advance(seconds)
            action = "Approche" if simulation.held_pile(plan.cup) is None else "Fin de l’approche"
            record(f"{action} de la {name}")

        if not _take_pickup(simulation, plan.pile, plan.cup, plan.pickup,
                            advance=approach_step, deadline=BUILD_DEADLINE):
            unavailable.add(plan.pile)
            continue
        advance(f"Prise de la {name}")
        delivered = all(move(target, f"Transport de la {name}") for target in plan.drop)
        if simulation.time >= BUILD_DEADLINE:
            break
        simulation.toggle_suction(plan.cup)
        advance(f"Dépôt de la {name}" if delivered else f"Transport bloqué · {name}")
        if delivered:
            def retreat_step(seconds):
                simulation.advance(seconds)
                record(f"Recul après le dépôt · {name}")

            _drop_retreat(simulation, plan.cup, advance=retreat_step, deadline=BUILD_DEADLINE)
            for target in plan.retreat:
                if not move(target, f"Recul après le dépôt · {name}"):
                    break
        else:
            unavailable.add(plan.pile)


