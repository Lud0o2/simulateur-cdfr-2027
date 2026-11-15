"""Géométrie pure : polygones, contacts balayés et durées de translation."""

from functools import lru_cache
from math import atan2, cos, hypot, sin, sqrt

from .constantes import CONTACT_GAP
from .modele import Point, Rectangle


def edges(points: tuple[Point, ...]):
    return zip(points, points[1:] + points[:1])


def rectangle_corners(rectangle: Rectangle) -> tuple[Point, ...]:
    left, top, right, bottom = rectangle
    return (left, top), (right, top), (right, bottom), (left, bottom)


def travel_time(distance: float, speed: float = 1000.0, acceleration: float = 1000.0) -> float:
    """Durée idéale d'une translation de l'arrêt à l'arrêt."""
    if distance < speed**2 / acceleration:
        return 2 * sqrt(distance / acceleration)
    return distance / speed + speed / acceleration


def contact_axes(corners, footprint):
    for (ax, ay), (bx, by) in (*edges(footprint), *edges(corners)):
        ax, ay = by - ay, ax - bx
        offsets = [px * ax + py * ay for px, py in footprint]
        bounds = [px * ax + py * ay for px, py in corners]
        yield ax, ay, min(offsets), max(offsets), min(bounds), max(bounds)


def swept_contact(position, delta, axes, *, include_boundary=False):
    x, y = position
    dx, dy = delta
    enter, leave = 0.0, 1.0
    for ax, ay, lo, hi, other_low, other_high in axes:
        center = x * ax + y * ay
        low, high = center + lo, center + hi
        speed = dx * ax + dy * ay
        if speed == 0:
            tolerance = CONTACT_GAP * hypot(ax, ay) if include_boundary else 0
            if high <= other_low - tolerance or low >= other_high + tolerance:
                return 1.0
            continue
        if speed > 0:
            first = (other_low - high) / speed
            last = (other_high - low) / speed
        else:
            first = (other_high - low) / speed
            last = (other_low - high) / speed
        if first > enter:
            enter = first
        if last < leave:
            leave = last
        if enter >= leave:
            return 1.0
    return enter


def polygon_contact(
    position: Point, delta: Point, corners: tuple[Point, ...], footprint: tuple[Point, ...],
    *, include_boundary: bool = False,
) -> float:
    """Renvoie la fraction du trajet où les deux polygones commencent à se chevaucher."""
    return swept_contact(position, delta, contact_axes(corners, footprint), include_boundary=include_boundary)


def table_contact(position: Point, delta: Point, footprint: tuple[Point, ...], *, width: float, height: float) -> float:
    x, y = position
    dx, dy = delta
    xs, ys = zip(*footprint)
    fraction = 1.0
    if dx < 0:
        fraction = min(fraction, (-min(xs) - x) / dx)
    if dx > 0:
        fraction = min(fraction, (width - max(xs) - x) / dx)
    if dy < 0:
        fraction = min(fraction, (-min(ys) - y) / dy)
    if dy > 0:
        fraction = min(fraction, (height - max(ys) - y) / dy)
    return fraction


def throne_room_score(position: Point, footprint: tuple[Point, ...], room: Rectangle) -> int:
    """Points de retour pour la projection d'un châssis à l'arrêt."""
    x, y = position
    left, top, right, bottom = room
    if all(left <= x + dx <= right and top <= y + dy <= bottom for dx, dy in footprint):
        return 10
    return 5 if polygon_contact(position, (0, 0), rectangle_corners(room), footprint) < 1 else 0


def arc_hits_edge(point: Point, angle: float, start: Point, end: Point) -> bool:
    """Teste l'intersection de l'arc d'un sommet avec un segment, autour de l'origine."""
    px, py = point
    sx, sy = start
    dx, dy = end[0] - sx, end[1] - sy
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return False
    closest = -(sx * dx + sy * dy) / length_squared
    nx, ny = sx + closest * dx, sy + closest * dy
    chord_squared = (px * px + py * py - nx * nx - ny * ny) / length_squared
    if chord_squared < 0:
        return False
    chord = sqrt(chord_squared)
    for fraction in (closest - chord, closest + chord):
        if 0 <= fraction <= 1:
            x, y = sx + fraction * dx, sy + fraction * dy
            swept = atan2(px * y - py * x, px * x + py * y)
            if 0 < swept <= angle or angle <= swept < 0:
                return True
    return False


@lru_cache(maxsize=4096)
def _polygon_bounds(polygon):
    xs, ys = zip(*polygon)
    return min(xs), min(ys), max(xs), max(ys)


@lru_cache(maxsize=16384)
def _rotated_vertices(vertices, angle):
    cosine, sine = cos(angle), sin(angle)
    return tuple((x * cosine - y * sine, x * sine + y * cosine) for x, y in vertices)


@lru_cache(maxsize=16384)
def _footprint_radius(footprint):
    return max(hypot(*vertex) for vertex in footprint)
