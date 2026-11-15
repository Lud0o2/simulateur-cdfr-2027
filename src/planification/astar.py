"""Routeur A* actuel sur un graphe de visibilité."""

from functools import lru_cache
from heapq import heappop, heappush
from itertools import pairwise
from math import ceil, floor, hypot, pi

from pymunk import Vec2d

from ..domaine.constantes import CONTACT_GAP
from ..domaine.geometrie import (
    _polygon_bounds as _bounds, contact_axes, rectangle_corners, swept_contact, travel_time,
)
from ..physique.constantes import DRIVE_ACCELERATION, PHYSICS_STEP, TURN_SPEED
from ..physique.moteur import Simulation
from ..terrains.officiel_2027 import TABLE_HEIGHT, TABLE_WIDTH
from .base import Approach
from .rotation import _pivot_clear, _turn_envelope


ROUTE_MARGIN = 5  # mm autour des obstacles pendant les trajets

WAYPOINTS = tuple(Vec2d(x, y) for x in (300, 600, 950, 1200, 1450, 1800, 2250, 2700)
                  for y in (260, 500, 750, 1000, 1250, 1500, 1740))
WAYPOINT_TIMES = {source: {target: travel_time(source.get_distance(target)) for target in WAYPOINTS}
                  for source in WAYPOINTS}
OPEN_SPACE = Vec2d(1450, 1000)


@lru_cache(maxsize=16384)
def _segment_time(start, end):
    return travel_time(start.get_distance(end))


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
    pivots = [(travel_time(start.get_distance(pivot)), travel_time(pivot.get_distance(end)), pivot)
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




class AstarPlanner:
    id = "astar"

    def planifier(self, demande):
        obstacles = tuple((obstacle, _bounds(obstacle)) for obstacle in demande.obstacles)
        approach = _approach(
            Vec2d(*demande.depart), Vec2d(*demande.arrivee), demande.angle_vise,
            demande.empreintes, obstacles,
        )
        if approach is None:
            return None
        from .base import Trajet

        return Trajet(approach.duration, approach.before, approach.turn, approach.after, approach.rolling)
