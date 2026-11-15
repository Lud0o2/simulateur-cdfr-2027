"""Match automatique sans fenêtre et lecture des essais enregistrés."""

import argparse
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count, pairwise
import json
from math import atan2, ceil, cos, copysign, dist, floor, hypot, pi, radians, remainder
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import perf_counter

from pymunk import Vec2d

from .simulateur import (
    CONTACT_GAP, DRIVE_ACCELERATION, PHYSICS_STEP, START, TABLE_HEIGHT, TABLE_WIDTH, THRONE_ROOM,
    TOWER_CENTERS, TOWER_RADIUS, TURN_SPEED, WALL_ZONES, Block, Simulation, _finite_number, _polygon_bounds as _bounds,
    construction_score, contact_axes, polygon_contact, rectangle_corners, scoring_zones,
    swept_contact, table_contact, throne_room_score, travel_time,
)


MATCH_SECONDS = 100
BUILD_DEADLINE = MATCH_SECONDS - 10
FRAME_STEP = 0.05
PICKUP_STANDOFF = 25  # mm entre la ventouse et le grand côté au départ
PICKUP_STROKE = 50  # mm vers la pile
PICKUP_GAP = 2  # mm entre la ventouse et le grand côté au moment de la prise
DROP_RETREAT = 50  # mm à l'opposé de la ventouse après la dépose
ROUTE_MARGIN = 5  # mm autour des obstacles pendant les trajets
REPLAY_DIR = Path(__file__).resolve().parents[1] / "replays"


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


# shortcut: points échantillonnés entre les rangées, à densifier pour les accès plus étroits.
WAYPOINTS = tuple(Vec2d(x, y) for x in (300, 600, 950, 1200, 1450, 1800, 2250, 2700)
                  for y in (260, 500, 750, 1000, 1250, 1500, 1740))
WAYPOINT_TIMES = {source: {target: travel_time(source.get_distance(target)) for target in WAYPOINTS}
                  for source in WAYPOINTS}
OPEN_SPACE = Vec2d(1450, 1000)


@lru_cache(maxsize=16384)
def _segment_time(start, end):
    return travel_time(start.get_distance(end))


@dataclass(frozen=True)
class Approach:
    duration: float
    before: tuple[Vec2d, ...]
    turn: float
    after: tuple[Vec2d, ...]
    rolling: bool = False

    def distance_from(self, start):
        return sum(a.get_distance(b) for a, b in pairwise((start, *self.before, *self.after)))


@dataclass(frozen=True)
class Pickup:
    approach: Approach
    end: Vec2d

    @property
    def duration(self):
        return self.approach.duration + travel_time(self.end.get_distance((self.approach.after or self.approach.before)[-1]))

    def distance_from(self, start):
        contact = (self.approach.after or self.approach.before)[-1]
        return self.approach.distance_from(start) + contact.get_distance(self.end)


@dataclass(frozen=True)
class Plan:
    pile: int
    cup: int
    pickup: Pickup
    drop: tuple[Vec2d, ...]
    retreat: tuple[Vec2d, ...]


def _clear(start, end, footprints, obstacles):
    return _clear_bounded(start, end, tuple((shape, _bounds(shape)) for shape in footprints), obstacles)


@lru_cache(maxsize=4096)
def _rotated_footprint(footprint, angle):
    return tuple(Vec2d(*corner).rotated(angle) for corner in footprint)


def _clear_bounded(start, end, footprints, obstacles, margin=0):
    delta = end - start
    min_x, max_x = sorted((start.x, end.x))
    min_y, max_y = sorted((start.y, end.y))
    for footprint, (left, top, right, bottom) in footprints:
        if not (-CONTACT_GAP <= min_x + left and max_x + right <= TABLE_WIDTH + CONTACT_GAP
                and -CONTACT_GAP <= min_y + top and max_y + bottom <= TABLE_HEIGHT + CONTACT_GAP):
            return False
        sweep_left, sweep_right = min_x + left, max_x + right
        sweep_top, sweep_bottom = min_y + top, max_y + bottom
        for corners, (x1, y1, x2, y2) in obstacles:
            if (sweep_left >= x2 + margin or sweep_right <= x1 - margin
                    or sweep_top >= y2 + margin or sweep_bottom <= y1 - margin):
                continue
            if swept_contact(start, delta, _planned_axes(corners, footprint, margin)) < 1:
                return False
    return True


@lru_cache(maxsize=4096)
def _planned_axes(corners, footprint, margin):
    return tuple((x, y, low, high, left - gap, right + gap)
                 for x, y, low, high, left, right in contact_axes(corners, footprint)
                 for gap in (margin * hypot(x, y),))


@lru_cache(maxsize=64)
def _visibility(footprints, obstacles, margin):
    footprints = tuple((shape, _bounds(shape)) for shape in footprints)

    @lru_cache(maxsize=4096)
    def visible(start, end):
        return _clear_bounded(start, end, footprints, obstacles, margin)

    return visible


@lru_cache(maxsize=256)
def _obstacle_waypoints(footprints, obstacles):
    points = []
    gap = ROUTE_MARGIN + 1
    for footprint in footprints:
        left, top, right, bottom = _bounds(footprint)
        for _, (x1, y1, x2, y2) in obstacles:
            # Coins du rectangle interdit au centre du châssis par cette paire de contours.
            blocked = (x1 - right - gap, y1 - bottom - gap, x2 - left + gap, y2 - top + gap)
            points.append((blocked, tuple(Vec2d(*point) for point in rectangle_corners(blocked))))
    return tuple(points)


def _corner_bypasses(start, end, corners):
    # Un seul point peut suffire pour passer du même côté des deux coins.
    delta = end - start
    for a, b in pairwise((*corners, corners[0])):
        for first, second in ((a, b), (b, a)):
            incoming, outgoing = first - start, second - end
            cross = incoming.cross(outgoing)
            if abs(cross) < CONTACT_GAP:
                continue
            along = delta.cross(outgoing) / cross
            back = delta.cross(incoming) / cross
            if along >= 1 and back >= 1:
                yield start + along * incoming


# Les empreintes et les obstacles font partie de la clé, y compris après un déplacement de pile.
@lru_cache(maxsize=4096)
def _route(start, end, footprints, obstacles, limit=float("inf")):
    """A* sur les segments libres, en minimisant le temps de translation."""
    direct = _segment_time(start, end)
    if direct > limit:
        return (), 0
    visible = _visibility(footprints, obstacles, ROUTE_MARGIN)
    if visible(start, end):
        return (end,), direct
    if not visible(start, start) or not visible(end, end):
        exact = _visibility(footprints, obstacles, 0)
        if not exact(start, start) or not exact(end, end):
            return (), 0
        # Autorise le premier ou dernier segment à rejoindre une pose de prise ou de dépôt serrée.
        close = {point for point in (start, end) if not visible(point, point)}
        with_margin = visible

        def visible(a, b):
            return exact(a, b) if a in close or b in close else with_margin(a, b)

        if visible(start, end):
            return (end,), direct
    left, right = sorted((start.x, end.x))
    top, bottom = sorted((start.y, end.y))
    nearby = [corners for (x1, y1, x2, y2), corners in _obstacle_waypoints(footprints, obstacles)
              if left <= x2 and right >= x1 and top <= y2 and bottom >= y1]
    points = dict.fromkeys((Vec2d(start.x, end.y), Vec2d(end.x, start.y), *WAYPOINTS))
    for corners in nearby:
        points.update(dict.fromkeys(corners))
    if len(nearby) == 1:
        points.update(dict.fromkeys(_corner_bypasses(start, end, nearby[0])))
    remaining = {end: 0.0}
    start_times = {}
    best = {start: 0.0}
    infinity = float("inf")
    # Un détour court donne une borne pour éviter de développer les branches plus lentes.
    ranked = []
    for point in points:
        if point == start or point == end:
            continue
        first, tail = _segment_time(start, point), _segment_time(point, end)
        if first + tail <= limit + CONTACT_GAP:
            ranked.append((first + tail, first, tail, point))
    for duration, first, tail, target in sorted(ranked):
        if duration > limit:
            break
        if not visible(target, target):
            continue
        start_times[target], remaining[target] = first, tail
        if visible(start, target) and visible(target, end):
            limit = duration
    pending = [(direct, 0.0, start, ())]
    while pending:
        _, duration, point, path = heappop(pending)
        if duration > best[point]:
            continue
        if point == end:
            return path, duration
        segment_times = start_times if point == start else WAYPOINT_TIMES.get(point, {})
        for target, tail in remaining.items():
            if target == end:
                segment = remaining.get(point, direct)
            else:
                segment = segment_times.get(target)
                if segment is None:
                    segment = _segment_time(point, target)
            cost = duration + segment
            estimate = cost + tail
            if cost >= best.get(target, infinity) or estimate > limit:
                continue
            if not visible(point, target):
                continue
            best[target] = cost
            heappush(pending, (estimate, cost, target, (*path, target)))
    return (), 0


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


@lru_cache(maxsize=4096)
def _approach(start, end, angle, footprints, obstacles) -> Approach | None:
    """Cherche un trajet avec une rotation en roulant ou un pivot dans un espace libre."""
    if not angle:
        path, duration = _route(start, end, footprints, obstacles)
        return Approach(duration, (), 0, path) if path else None

    polygons = tuple(corners for corners, _ in obstacles)
    rotated = tuple(_rotated_footprint(shape, angle) for shape in footprints)
    clear_after = _visibility(rotated, obstacles, 0)
    if not clear_after(end, end):
        return None
    best = None
    turn_time = abs(angle) / TURN_SPEED
    if 0 < abs(angle) < pi:
        distance = start.get_distance(end)
        # Un pas de marge couvre l'arrondi de la rotation au pas physique.
        turning_time = turn_time + PHYSICS_STEP
        turning_distance = DRIVE_ACCELERATION * turning_time * (turning_time + PHYSICS_STEP) / 2
        if distance >= turning_distance:
            turning_end = start + (end - start) * (turning_distance / distance)
            envelopes = tuple(_turn_envelope(shape, final, angle)
                              for shape, final in zip(footprints, rotated, strict=True))
            if (_visibility(envelopes, obstacles, ROUTE_MARGIN)(start, turning_end)
                    and _visibility(rotated, obstacles, ROUTE_MARGIN)(turning_end, end)):
                return Approach(travel_time(distance), (end,), angle, (), rolling=True)
    # shortcut: un seul changement de cap par trajet, à étendre pour les passages plus étroits.
    along = tuple(start + (end - start) * fraction for fraction in (0.25, 0.5, 0.75))
    if 0 < start.get_distance(end) < 150:
        normal = (end - start).perpendicular().normalized()
        middle = (start + end) / 2
        along += tuple(middle + normal * shift for shift in (40, -40, 80, -80))
    pivots = (start, end, *along, OPEN_SPACE, *WAYPOINTS)
    pivots = [(_segment_time(start, pivot), _segment_time(pivot, end), pivot)
              for pivot in dict.fromkeys(pivots)]
    clear_before = _visibility(footprints, obstacles, 0)
    for first_direct, last, pivot in sorted(pivots, key=lambda p: p[0] + p[1]):
        remaining = best.duration - turn_time if best else float("inf")
        # Une petite marge protège les départages aux erreurs d'arrondi.
        limit = remaining + CONTACT_GAP
        if first_direct + last > limit:
            break
        if (not clear_before(pivot, pivot) or not clear_after(pivot, pivot)
                or not _pivot_clear(pivot, footprints, angle, polygons)):
            continue
        before, first = _route(start, pivot, footprints, obstacles, limit - last)
        if not before:
            continue
        after, second = _route(pivot, end, rotated, obstacles, limit - first)
        duration = first + second + turn_time
        if after and (best is None or duration < best.duration):
            best = Approach(duration, before, angle, after)
    return best


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


def _pickup_options(simulation, block, cup, state):
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
        approach = _approach(start, standoff, angle, footprints, obstacles)
        if approach is not None:
            approach = replace(approach, duration=approach.duration + travel_time(standoff.get_distance(contact)),
                               after=(*approach.after, contact))
            end = standoff - normal * PICKUP_STROKE
            held = tuple(Vec2d(*corner) - contact for corner in block.corners)
            others = tuple(obstacle for obstacle in obstacles if obstacle[0] != block.corners)
            yield Pickup(approach, _stroke_end(contact, end - contact, (*rotated, held), others))


def _pickup_approach(simulation, block, cup, state):
    return min(_pickup_options(simulation, block, cup, state), key=lambda option: option.duration, default=None)


def _wall_drop_zones(block: Block):
    center = Vec2d(*block.position)
    if any(center.get_distance(tower) <= TOWER_RADIUS + CONTACT_GAP for tower in TOWER_CENTERS):
        return (), False
    walls, _ = scoring_zones(block)
    return walls, bool(walls)


def _drop_retreat(simulation, cup, *, advance=None, deadline=None):
    start, footprints, obstacles = _action_state(simulation)
    _, facing = simulation.geometry.mounts[cup - 1]
    end = _stroke_end(start, -DROP_RETREAT * facing.rotated(simulation.heading), footprints, obstacles)
    _follow(simulation, Approach(travel_time(start.get_distance(end)), (), 0, (end,)),
            advance=advance, deadline=deadline)


def _drop_target(block, destination, angle, start):
    center = Vec2d(*block.position)
    turn = remainder(remainder(angle, 2 * pi) - block.angle, 2 * pi)
    return destination - (center - start).rotated(turn), turn


def _drop_approach(block, destination, angle, state):
    start, footprints, obstacles = state
    end, turn = _drop_target(block, destination, angle, start)
    center = Vec2d(*block.position)
    corners = tuple(tuple(destination + (Vec2d(*corner) - center).rotated(turn)) for corner in block.corners)
    if not _wall_drop_zones(Block(tuple(destination), block.angle + turn, corners))[1]:
        return None
    return _approach(start, end, turn, footprints, obstacles)


def place_pile(simulation: Simulation, cup: int, x: float, y: float, angle: float, *, advance=None,
               deadline=None, on_release=None) -> bool:
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
    approach = _drop_approach(block, destination, angle, _action_state(simulation))
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


def _frame(simulation, label, *, score=None):
    def point(coords):
        return [round(value, 3) for value in coords]

    movement = simulation.movement
    blocks = simulation.blocks
    return {
        "t": round(simulation.time, 6),
        "position": point(simulation.position),
        "heading": round(simulation.heading, 6),
        "robot": [point(corner) for corner in simulation.footprint],
        "piles": [[point(corner) for corner in block.corners] for block in blocks],
        "cups": [[point(cup.position), cup.enabled] for cup in simulation.suction_cups],
        "target": point(movement.end) if simulation.moving and not movement.turn else None,
        "blocked": bool(movement and (movement.blocked or movement.rotation_blocked)),
        "score": simulation.score_for_blocks(blocks) if score is None else score,
        "action": label,
    }


def _play_batches(simulation, record, policy, drop_poses):
    """Charge les ventouses disponibles et dépose les piles une par une."""
    def perform(action, label, **choice):
        def advance(seconds):
            simulation.advance(seconds)
            record(label)

        return action(simulation, **choice, advance=advance, deadline=BUILD_DEADLINE)

    cups = range(1, len(simulation.geometry.mounts) + 1)
    while simulation.time < BUILD_DEADLINE:
        held = {}
        missed = set()
        while len(held) < len(cups) and simulation.time < BUILD_DEADLINE:
            blocks = simulation.blocks
            carried = set(held.values())
            free_cups = tuple(cup for cup in cups if cup not in held)
            choices = []
            approaches = {}
            state = _action_state(simulation)
            for index, block in enumerate(blocks):
                if index in carried or scoring_zones(block)[1]:
                    continue
                for cup in free_cups:
                    if (index, cup) in missed:
                        continue
                    approach = _pickup_approach(simulation, block, cup, state)
                    if approach is not None:
                        approaches[index, cup] = approach
                        choices.append((approach.duration, index, cup))
            if not choices:
                break
            choices.sort()
            if policy is not None:
                choices = policy.order(simulation, held, choices, pickup=True, can_finish=True, pickup_plans=approaches)
                choice = choices[0]
            else:
                # À un pas de simulation du match près, privilégier le plus petit changement de cap.
                fastest = choices[0][0]
                choice = min((item for item in choices if item[0] <= fastest + FRAME_STEP),
                             key=lambda item: (abs(approaches[item[1:]].approach.turn), item[2], item[0]))
            if choice is None:
                policy.observe(choice)
                if held:
                    break
                record("Fin des constructions · Retour demandé")
                return
            _, index, cup = choice

            if not perform(_take_pickup, f"Prise pile {index + 1} · ventouse {cup}",
                           pile=index, cup=cup, pickup=approaches[index, cup]):
                missed.add((index, cup))
                continue
            if policy is not None:
                policy.observe(choice)
            held[cup] = index
            missed.clear()
            record(f"Pile {index + 1} tenue · {len(held)} pile(s) en charge")
        if not held:
            break

        while held and simulation.time < BUILD_DEADLINE:
            blocks = simulation.blocks
            sites = tuple(scoring_zones(block) for index, block in enumerate(blocks) if index not in held.values())
            score = construction_score(sites)
            start, _, obstacles = _action_state(simulation)
            placements = []
            for cup, index in held.items():
                block = blocks[index]
                center = Vec2d(*block.position)
                footprint = tuple(Vec2d(*corner) - center for corner in block.corners)
                for site, angle in drop_poses:
                    axis_turn = remainder(angle - block.angle, 2 * pi)
                    relative = _rotated_footprint(footprint, axis_turn)
                    corners = tuple(tuple(site + corner) for corner in relative)
                    projected = Block(tuple(site), angle, corners)
                    zones = _wall_drop_zones(projected)
                    if not zones[1] or not _clear(site, site, (relative,), obstacles):
                        continue
                    gain = construction_score((*sites, zones)) - score
                    # À gain égal, éviter de remplir d'abord le passage central des remparts.
                    central_penalty = max(0, 650 - abs(site.y - TABLE_HEIGHT / 2)) / 400
                    # Le rectangle garde la même pose lorsque le châssis l'aborde dans l'autre sens.
                    for offset in ((0,) if policy is None else (0, pi)):
                        turn = axis_turn if not offset else remainder(angle + offset - block.angle, 2 * pi)
                        distance = start.get_distance(site - (center - start).rotated(turn))
                        duration = travel_time(distance) + abs(turn) / TURN_SPEED + central_penalty
                        placements.append((gain, duration, cup, site, angle + offset))
            placements.sort(key=lambda item: (-item[0], item[1]))
            if policy is not None and placements:
                placements = policy.order(simulation, held, placements, pickup=False)
            for choice in placements:
                _, _, cup, site, angle = choice
                index = held[cup]

                def advance_deposit(seconds):
                    simulation.advance(seconds)
                    action = "Recul après dépôt" if simulation.held_pile(cup) is None else "Transport"
                    record(f"{action} pile {index + 1} · ventouse {cup}")

                def record_deposit():
                    record(f"Dépôt pile {index + 1} · {len(held) - 1} pile(s) en charge")

                angles = (angle, angle + pi) if policy is None else (angle,)
                if any(place_pile(simulation, cup, site.x, site.y, drop_angle, advance=advance_deposit,
                                  deadline=BUILD_DEADLINE, on_release=record_deposit)
                       for drop_angle in angles if simulation.time < BUILD_DEADLINE):
                    if policy is not None:
                        policy.observe(choice)
                    held.pop(cup)
                    break
                if simulation.time >= BUILD_DEADLINE:
                    break
            else:
                return


def _return_home(simulation, record, *, complete_only=False, shortest=False):
    deadline = MATCH_SECONDS - FRAME_STEP  # Une arrivée au buzzer doit déjà être à l'arrêt.
    if simulation.time >= deadline:
        return
    if simulation.moving:
        simulation.move_to(*simulation.position)
    current = simulation.return_score
    if current == 10:
        return

    start, footprints, obstacles = _action_state(simulation)
    _, top, right, bottom = THRONE_ROOM
    targets = {10: [], 5: []}
    # shortcut: cinq poses par cap, à élargir si les accès de la salle sont encombrés.
    turns = dict.fromkeys((0, *(remainder(heading - simulation.heading, 2 * pi)
                              for heading in (0, pi / 2, pi, -pi / 2))))
    for turn in turns:
        hull = tuple(Vec2d(*corner).rotated(turn) for corner in footprints[0])
        left_edge, _, right_edge, _ = _bounds(hull)
        for target in (Vec2d(*START), Vec2d(right - right_edge - 10, START[1]),
                       Vec2d(right - left_edge - 20, START[1]), Vec2d(START[0], top + 50),
                       Vec2d(START[0], bottom - 50)):
            score = throne_room_score(target, hull)
            if score > current:
                targets[score].append((target, turn))

    def advance(seconds):
        simulation.advance(seconds)
        record("Retour à la salle du trône")

    for score in ((10,) if complete_only else (10, 5)):
        best = None
        best_cost = float("inf")
        for target, turn in targets[score]:
            lower_bound = start.get_distance(target) if shortest else _segment_time(start, target)
            if lower_bound >= best_cost:
                continue
            route = _approach(start, target, turn, footprints, obstacles)
            if route is None or simulation.time + route.duration > deadline:
                continue
            cost = route.distance_from(start) if shortest else route.duration
            if cost < best_cost:
                best, best_cost = route, cost
        if best is not None:
            if (_follow(simulation, best, advance=advance, deadline=deadline)
                    and current == 0 and simulation.return_score == 5):
                _return_home(simulation, record, complete_only=complete_only, shortest=shortest)
            return


def _collection_choices(simulation, cups, held, *, nearest=None):
    state = _action_state(simulation)
    start = state[0]
    carried = set(held.values())
    stocks = [(index, block) for index, block in enumerate(simulation.blocks) if index not in carried]
    if nearest is not None:
        stocks = sorted(stocks, key=lambda stock: dist(start, stock[1].position))[:nearest]
    available = tuple(cup for cup in cups if cup not in held)
    choices = []
    for index, block in stocks:
        for cup in available:
            for pickup in _pickup_options(simulation, block, cup, state):
                choices.append((pickup.distance_from(start), pickup.duration, index, cup, pickup))
    choices.sort(key=lambda choice: choice[:2])
    return choices


def _paired_cups(geometry):
    paired = {}
    for direction in (1, -1):
        face = sorted((mount[1], cup) for cup, (mount, facing) in enumerate(geometry.mounts, 1)
                      if facing.x * direction > 1 - CONTACT_GAP)
        if len(face) == 2:
            (_, left), (_, right) = face
            paired[left] = right, 1
            paired[right] = left, -1
    return paired


def _collection_actions(simulation, cups, held, paired, pair):
    choices = _collection_choices(simulation, cups, held, nearest=3)
    if pair is not None:
        pile, cup, heading = pair
        return [(choice, None) for choice in choices if choice[2:4] == (pile, cup)
                and cos(simulation.heading + choice[4].approach.turn - heading) > 0]

    blocks = simulation.blocks
    carried = set(held.values())
    actions = []
    for choice in choices:
        _, _, index, cup, pickup = choice
        if cup not in paired:
            actions.append((choice, None))
            continue
        other_cup, sign = paired[cup]
        if other_cup in held:
            continue
        block = blocks[index]
        heading = simulation.heading + pickup.approach.turn
        for other, neighbor in enumerate(blocks):
            if other == index or other in carried:
                continue
            offset = (Vec2d(*neighbor.position) - block.position).rotated(-heading)
            lateral = sign * offset.y
            # Les centres voisins sont espacés de 500 mm sur les rangées officielles.
            if (0 < lateral <= 550 and abs(offset.x) <= 55
                    and abs(remainder(neighbor.angle - block.angle, pi)) <= radians(10)):
                actions.append((choice, (other, other_cup, heading)))
    return actions


def _collection_trip(simulation, cups, load):
    best = None
    best_distance = float("inf")
    # shortcut: borne les essais de prise par tournée, à relever si les chargements restent bloqués.
    attempts = 48

    def search(state, picks, distance):
        nonlocal best, best_distance, attempts
        if distance >= best_distance:
            return
        held = {cup: index for cup in cups if (index := state.held_pile(cup)) is not None}
        if len(held) == load:
            position = state.position

            def record_return(_label):
                nonlocal position, distance
                distance += dist(position, state.position)
                position = state.position

            _return_home(state, record_return, complete_only=True, shortest=True)
            if state.return_score == 10 and round(state.time, 6) < MATCH_SECONDS and distance < best_distance:
                best, best_distance = picks, distance
            return

        if attempts == 0:
            return
        for _, _, index, cup, pickup in _collection_choices(state, cups, held):
            if attempts == 0:
                break
            attempts -= 1
            candidate = deepcopy(state)
            travelled = 0.0

            def advance(seconds):
                nonlocal travelled
                position = candidate.position
                candidate.advance(seconds)
                travelled += dist(position, candidate.position)

            if _take_pickup(candidate, index, cup, pickup, advance=advance, deadline=MATCH_SECONDS):
                search(candidate, (*picks, (index, cup, pickup)), distance + travelled)

    search(deepcopy(simulation), (), 0.0)
    return best


def play_collection(simulation: Simulation, *, policy=None, record_frames=True):
    """Charge les ventouses disponibles et ramène les piles au départ."""
    total = len(simulation.blocks)
    cups = range(1, len(simulation.geometry.mounts) + 1)
    paired = _paired_cups(simulation.geometry)
    delivered = 0
    frames = []

    def record(label):
        if not record_frames:
            return
        frame = _frame(simulation, label, score=delivered)
        frame["t"] = min(frame["t"], MATCH_SECONDS)
        if frames and frames[-1]["t"] == frame["t"]:
            frames[-1] = frame
        else:
            frames.append(frame)

    record("Départ")
    while delivered < total and round(simulation.time, 6) < MATCH_SECONDS:
        load = min(len(cups), total - delivered)
        held = {}
        pair = None
        if policy is None:
            plan = _collection_trip(simulation, cups, load)
            if plan is None:
                break
        for step in range(load):
            if policy is None:
                index, cup, pickup = plan[step]
            else:
                actions = _collection_actions(simulation, cups, held, paired, pair)
                if not actions:
                    break
                (_, _, index, cup, pickup), pair = policy(simulation, held, actions)

            def advance(seconds):
                simulation.advance(seconds)
                record(f"Prise · ventouse {cup}")

            if not _take_pickup(simulation, index, cup, pickup, advance=advance, deadline=MATCH_SECONDS):
                break
            held[cup] = index
        else:
            _return_home(simulation, record, complete_only=True, shortest=policy is None)
            unloaded = simulation.unload_at_start() if round(simulation.time, 6) < MATCH_SECONDS else 0
            if unloaded:
                delivered += unloaded
                record(f"Décharge · {delivered}/{total} piles ramenées")
                continue
        break

    if delivered < total:
        record(f"Fin · {delivered}/{total} piles ramenées")
    replay = {"strategie": "opti_take", "obstacles": simulation.obstacles, "frames": frames,
              "score": delivered, "total_piles": total, "duree": min(round(simulation.time, 6), MATCH_SECONDS)}
    if record_frames:
        replay["distance_mm"] = sum(dist(a["position"], b["position"]) for a, b in pairwise(frames))
    return replay


def play_match(simulation: Simulation, *, batches=False, policy=None, record_frames=True):
    batches = batches or policy is not None
    frames = []

    def record(label, bonus=None):
        if not record_frames:
            return
        if frames and simulation.time == 0 and bonus is None:
            return
        frame = _frame(simulation, label)
        frame["score"] += bonus or 0
        if frames and frames[-1]["t"] == frame["t"]:
            if batches and bonus is None and frames[-1]["action"].startswith(("Pile ", "Dépôt ")):
                return
            frames[-1] = frame
        else:
            frames.append(frame)

    record("Départ")
    offsets = getattr(policy, "drop_offsets", None)
    angles = getattr(policy, "drop_angles", ALIGNED_ANGLES)
    if batches:
        angles = tuple(min(DROP_ANGLES, key=lambda candidate: abs(remainder(candidate - angle, pi)))
                       for angle in angles)
        drop_poses = _shifted_drops(offsets, angles) if offsets else CENTERED_DROPS
        _play_batches(simulation, record, policy, tuple(dict.fromkeys((*drop_poses, *ALTERNATE_DROPS))))
    else:
        _play_reference(simulation, record)

    _return_home(simulation, record)
    bonus = simulation.return_score
    record(f"Fin du match · Retour +{bonus}" if bonus else "Fin du match", bonus)
    replay = {"obstacles": simulation.obstacles, "frames": frames,
              "score": simulation.score + bonus, "score_retour": bonus, "duree": round(simulation.time, 6)}
    if offsets and any(dx or dy for dx, dy in offsets):
        replay["decalages_depot_mm"] = offsets
    if any(angles):
        replay["rotations_depot_rad"] = angles
    return replay


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


def _trials(directory: Path):
    trials = {}
    for path in directory.glob("essai-*.json"):
        match = re.fullmatch(r"essai-([0-9]+)(?:--?[0-9]+(?:pts|piles)-[0-9]+\.[0-9]{2}s)?", path.stem)
        if match and path.is_file():
            trials[int(match[1])] = path
    return trials


def save_replay(replay, directory: Path = REPLAY_DIR, *, descriptive=False):
    unit = "piles" if replay.get("strategie") == "opti_take" else "pts"
    suffix = f"-{replay['score']:d}{unit}-{replay['duree']:.2f}s" if descriptive else ""
    directory.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(dir=directory, prefix=".essai-", suffix=".tmp", encoding="utf-8",
                              mode="w", delete=False)
    temporary = Path(file.name)
    try:
        with file:
            json.dump(replay, file, ensure_ascii=False, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        for number in count(max(_trials(directory), default=0) + 1):
            stem = f"essai-{number:04d}"
            reservation = directory / f".{stem}.lock"
            try:
                reservation.mkdir()
            except FileExistsError:
                continue
            try:
                # Deux sauvegardes peuvent viser le même numéro avec des noms différents.
                if (directory / f"{stem}.json").exists() or any(directory.glob(f"{stem}-*.json")):
                    continue
                path = directory / f"{stem}{suffix}.json"
                # Sous Windows, rename échoue si la destination existe. Sous POSIX, le lien offre cette garantie.
                if os.name == "nt":
                    temporary.rename(path)
                else:
                    path.hardlink_to(temporary)
                return number, path
            except FileExistsError:
                continue
            finally:
                reservation.rmdir()
    finally:
        temporary.unlink(missing_ok=True)


class Replay:
    def __init__(self, data):
        def coordinates(values, count):
            return (isinstance(values, (list, tuple)) and len(values) == count
                    and all(map(_finite_number, values)))

        def polygon(values, minimum):
            return (isinstance(values, list) and len(values) >= minimum
                    and all(coordinates(point, 2) for point in values))

        if not isinstance(data, dict) or not isinstance(data.get("frames"), list) or not data["frames"]:
            raise ValueError("le replay doit contenir au moins une image")
        obstacles = data.get("obstacles")
        if not isinstance(obstacles, (list, tuple)) or not all(coordinates(rect, 4) for rect in obstacles):
            raise ValueError("liste d’obstacles manquante ou invalide")
        self.frames = data["frames"]
        self.obstacles = obstacles
        self.collection = data.get("strategie") == "opti_take"
        self.times = []
        for index, frame in enumerate(self.frames, 1):
            if not isinstance(frame, dict):
                raise ValueError(f"image {index} invalide")
            for key in ("t", "position", "heading", "robot", "piles", "cups", "target", "blocked", "score", "action"):
                if key not in frame:
                    raise ValueError(f"image {index} : champ « {key} » manquant")

            timestamp = frame["t"]
            if not _finite_number(timestamp) or timestamp < 0 or (self.times and timestamp <= self.times[-1]):
                raise ValueError(f"image {index} : horodatage invalide")
            if not (coordinates(frame["position"], 2) and _finite_number(frame["heading"])
                    and (frame["target"] is None or coordinates(frame["target"], 2))
                    and polygon(frame["robot"], 3)
                    and isinstance(frame["piles"], list) and all(polygon(pile, 4) for pile in frame["piles"])
                    and isinstance(frame["cups"], list) and all(
                        isinstance(cup, list) and len(cup) == 2 and coordinates(cup[0], 2) and type(cup[1]) is bool
                        for cup in frame["cups"]
                    )
                    and type(frame["blocked"]) is bool and type(frame["score"]) is int
                    and isinstance(frame["action"], str)):
                raise ValueError(f"image {index} : données d’affichage invalides")
            if index > 1:
                previous = self.frames[index - 2]
                unloaded = frame["score"] - previous["score"] if self.collection else 0
                if (unloaded < 0 or len(frame["piles"]) != len(previous["piles"]) - unloaded
                        or len(frame["cups"]) != len(previous["cups"])):
                    raise ValueError(f"image {index} : nombre de piles ou de ventouses incohérent")
            self.times.append(timestamp)
        self.start = self.time = self.times[0]
        self.duration = self.times[-1]

    @property
    def frame(self):
        return self.frames[max(0, bisect_right(self.times, self.time) - 1)]


def completion_beep():
    if os.name == "nt":
        import winsound

        try:
            winsound.Beep(880, 180)
            return
        except RuntimeError:
            pass
    print("\a", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("collecte", help="Ramène les piles au départ et enregistre le replay.")
    player = commands.add_parser("replay", help="Ouvre le dernier essai ou le numéro indiqué.")
    player.add_argument("number", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "replay":
        trials = _trials(REPLAY_DIR)
        number = args.number if args.number is not None else max(trials, default=0)
        if number not in trials:
            parser.error(f"Essai {number} introuvable." if args.number is not None else "Aucun essai enregistré.")
        from .interface import create_window

        try:
            with trials[number].open(encoding="utf-8") as file:
                replay = Replay(json.load(file))
        except (OSError, ValueError, OverflowError) as error:
            parser.error(f"Impossible de lire l'essai {number} ({error}).")
        create_window(replay=replay, trial=number).mainloop()
    else:
        collection = args.command == "collecte"
        try:
            start = perf_counter()
            replay = (play_collection if collection else play_match)(Simulation())
            elapsed = perf_counter() - start
            number, _ = save_replay(replay, descriptive=collection)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        result = (f"{replay['score']}/{replay['total_piles']} piles" if collection else f"{replay['score']} points")
        distance = f" · {replay['distance_mm'] / 1000:.2f} m parcourus" if collection else ""
        print(f"Essai {number} · {result} en {replay['duree']:.2f} s simulées "
              f"(calcul : {elapsed:.2f} s){distance}", flush=True)
        completion_beep()


if __name__ == "__main__":
    main()
