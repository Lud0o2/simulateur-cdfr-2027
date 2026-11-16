from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from math import atan2, cos, hypot, isfinite, pi, radians, remainder, sin, sqrt
from pathlib import Path
import tomllib

import pymunk


Point = tuple[float, float]
Rectangle = tuple[float, float, float, float]
TABLE_WIDTH = 3000
TABLE_HEIGHT = 2000
SPEED = 1000  # mm/s
DRIVE_ACCELERATION = 1000  # mm/s² en translation
TURN_STEP = radians(15)
TURN_SPEED = radians(360)
TURN_ACCELERATION = radians(1200)  # rad/s², effort de pivot provisoire
ROBOT_FILE = Path(__file__).resolve().parents[1] / "RobotRobotronik.toml"
START = (250.0, 1000.0)
THRONE_ROOM = (0, 750, 500, 1250)  # Règlement Eurobot 2027, § D.3 et plan G.1.
TABLE = ((0, 0), (TABLE_WIDTH, 0), (TABLE_WIDTH, TABLE_HEIGHT), (0, TABLE_HEIGHT))
# Règlement Eurobot 2027, § D.4 et plan G.1 (X depuis le côté jaune).
START_BLOCKS = (
    *((x - 160, y, x + 160, y + 110) for y in (0, 1890) for x in (750, 1250, 1750, 2250)),
    (1445, 340, 1555, 660),
    (1445, 1340, 1555, 1660),
)
CONTACT_GAP = 1e-6  # mm, évite les chevauchements dus aux arrondis
PHYSICS_STEP = 1 / 480  # contacts plus stables quand une pile est coincée
ROBOT_MASS = 5.0  # kg, masse effective provisoire pour les contacts
PILE_MASS = 0.09
FLOOR_DECELERATION = 4200  # mm/s²
FLOOR_ANGULAR_DECELERATION = 10  # rad/s²
SUCTION_REACH = 5  # mm
SUCTION_ALIGNMENT = cos(radians(20))


def edges(points: tuple[Point, ...]):
    return zip(points, points[1:] + points[:1])


@dataclass(frozen=True, slots=True)
class RobotGeometry:
    contour: tuple[Point, ...]
    mounts: tuple[tuple[Point, pymunk.Vec2d], ...]


def _finite_number(value):
    return type(value) in (int, float) and isfinite(value)


def _read_point(value, field: str) -> Point:
    if not isinstance(value, list) or len(value) != 2 or not all(map(_finite_number, value)):
        raise ValueError(f"{field} doit contenir deux coordonnées finies en mm.")
    return value[0], value[1]


@cache
def load_robot(path: Path = ROBOT_FILE) -> RobotGeometry:
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
        if data.keys() - {"contour_mm", "ventouses"}:
            raise ValueError("Les champs autorisés sont contour_mm et ventouses.")
        points = data.get("contour_mm")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("contour_mm doit contenir au moins trois sommets.")
        contour = tuple(_read_point(point, f"contour_mm, sommet {index}")
                        for index, point in enumerate(points, 1))
        if any(not (0 <= START[0] + x <= TABLE_WIDTH and 0 <= START[1] + y <= TABLE_HEIGHT)
               for x, y in contour):
            raise ValueError("Le contour dépasse de la table à la position de départ.")
        vertices = tuple(pymunk.Vec2d(*point) for point in contour)
        turns = [(b - a).cross(point - a) for a, b in edges(vertices)
                 for point in vertices if point != a and point != b]
        if len(set(contour)) != len(contour) or not (
            all(isfinite(turn) and turn > 0 for turn in turns)
            or all(isfinite(turn) and turn < 0 for turn in turns)
        ):
            raise ValueError("contour_mm doit décrire un contour convexe, sans sommets alignés, dans l’ordre du périmètre.")

        cups = data.get("ventouses", [])
        if not isinstance(cups, list):
            raise ValueError("Déclare les ventouses avec des blocs [[ventouses]].")
        mounts = []
        for number, cup in enumerate(cups, 1):
            if not isinstance(cup, dict) or cup.keys() != {"position_mm", "angle_deg"}:
                raise ValueError(f"La ventouse {number} doit définir position_mm et angle_deg.")
            mount = _read_point(cup["position_mm"], f"Ventouse {number}, position_mm")
            angle = cup["angle_deg"]
            if not _finite_number(angle):
                raise ValueError(f"Ventouse {number}, angle_deg doit être un angle fini en degrés.")
            mounts.append((mount, pymunk.Vec2d(1, 0).rotated(radians(angle % 360))))
        return RobotGeometry(contour, tuple(mounts))
    except (OSError, ValueError, OverflowError) as error:
        raise ValueError(f"Configuration du robot {path.name} : {error}") from error


def rectangle_corners(rectangle: Rectangle) -> tuple[Point, ...]:
    left, top, right, bottom = rectangle
    return (left, top), (right, top), (right, bottom), (left, bottom)


def travel_time(distance: float) -> float:
    """Durée idéale d'une translation de l'arrêt à l'arrêt."""
    if distance < SPEED**2 / DRIVE_ACCELERATION:
        return 2 * sqrt(distance / DRIVE_ACCELERATION)
    return distance / SPEED + SPEED / DRIVE_ACCELERATION


def _wall_zone(x: float, y: float, angle: float = 0) -> tuple[Point, ...]:
    return tuple(tuple(pymunk.Vec2d(x, y) + pymunk.Vec2d(*vertex).rotated(radians(angle)))
                 for vertex in rectangle_corners((-200, -100, 200, 100)))


# Remparts jaunes, § D.4 et G.1. Origine au coin supérieur gauche du tapis.
WALL_ZONES = (_wall_zone(250, 400), _wall_zone(675, 575, 45), _wall_zone(850, 1000, 90),
              _wall_zone(675, 1425, -45), _wall_zone(250, 1600))
TOWER_CENTERS = ((500, 400), (850, 750), (850, 1250), (500, 1600))
TOWER_RADIUS = 100


def _world_corners(shape: pymunk.Poly) -> tuple[Point, ...]:
    return tuple(tuple(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices())


@lru_cache(maxsize=4096)
def _polygon_bounds(polygon):
    xs, ys = zip(*polygon)
    return min(xs), min(ys), max(xs), max(ys)


_BOUNDED_WALL_ZONES = tuple((zone, _polygon_bounds(zone)) for zone in WALL_ZONES)


def polygon_contact(
    position: Point, delta: Point, corners: tuple[Point, ...], footprint: tuple[Point, ...],
    *, include_boundary: bool = False,
) -> float:
    """Renvoie la fraction du trajet où les deux polygones commencent à se chevaucher."""
    return swept_contact(position, delta, contact_axes(corners, footprint), include_boundary=include_boundary)


def throne_room_score(position: Point, footprint: tuple[Point, ...]) -> int:
    """Points de retour pour la projection d'un châssis à l'arrêt."""
    x, y = position
    left, top, right, bottom = THRONE_ROOM
    if all(left <= x + dx <= right and top <= y + dy <= bottom for dx, dy in footprint):
        return 10
    return 5 if polygon_contact(position, (0, 0), rectangle_corners(THRONE_ROOM), footprint) < 1 else 0


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


def table_contact(position: Point, delta: Point, footprint: tuple[Point, ...]) -> float:
    x, y = position
    dx, dy = delta
    xs, ys = zip(*footprint)
    fraction = 1.0
    if dx < 0:
        fraction = min(fraction, (-min(xs) - x) / dx)
    if dx > 0:
        fraction = min(fraction, (TABLE_WIDTH - max(xs) - x) / dx)
    if dy < 0:
        fraction = min(fraction, (-min(ys) - y) / dy)
    if dy > 0:
        fraction = min(fraction, (TABLE_HEIGHT - max(ys) - y) / dy)
    return fraction


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


@lru_cache(maxsize=16384)
def _rotated_footprint(footprint, angle):
    return tuple(pymunk.Vec2d(*corner).rotated(angle) for corner in footprint)


@lru_cache(maxsize=16384)
def _footprint_radius(footprint):
    return max(hypot(*vertex) for vertex in footprint)


@dataclass(frozen=True)
class Movement:
    end: Point
    blocked: bool
    turn: bool = False
    rotation_blocked: bool = False


@dataclass(frozen=True)
class Block:
    position: Point
    angle: float
    corners: tuple[Point, ...]


@dataclass(frozen=True)
class SuctionCup:
    position: Point
    enabled: bool


@dataclass(frozen=True)
class _Grip:
    cup: int
    shape: pymunk.Poly
    angle: float


@dataclass
class _Pile:
    body: pymunk.Body
    shape: pymunk.Poly
    floor: tuple[pymunk.Constraint, pymunk.Constraint]
    size: Point
    grip: _Grip | None = None
    _block: Block | None = None

    def free_block(self) -> Block:
        position, angle = tuple(self.body.position), self.body.angle
        if self._block is None or self._block.position != position or self._block.angle != angle:
            self._block = Block(position, angle, _world_corners(self.shape))
        return self._block


@lru_cache(maxsize=16384)
def scoring_zones(block: Block) -> tuple[tuple[int, ...], bool]:
    left, top, right, bottom = _polygon_bounds(block.corners)
    # La tolérance sur les axes obliques peut dépasser CONTACT_GAP une fois projetée sur x ou y.
    gap = 2 * CONTACT_GAP
    walls = tuple(index for index, (zone, (x1, y1, x2, y2)) in enumerate(_BOUNDED_WALL_ZONES)
                  if x1 <= right + gap and left <= x2 + gap and y1 <= bottom + gap and top <= y2 + gap
                  and polygon_contact((0, 0), (0, 0), zone, block.corners, include_boundary=True) < 1)
    if walls:
        return walls, True
    origin = pymunk.Vec2d(*block.position)
    half_x, half_y = (pymunk.Vec2d(*block.corners[0]) - origin).rotated(-block.angle)
    for center in TOWER_CENTERS:
        x, y = (pymunk.Vec2d(*center) - origin).rotated(-block.angle)
        dx = max(0, abs(x) - abs(half_x))
        dy = max(0, abs(y) - abs(half_y))
        if dx * dx + dy * dy <= (TOWER_RADIUS + CONTACT_GAP) ** 2:
            return (), True
    return (), False


def free_wall_slots(blocks: Iterable[Block]) -> int:
    occupied = {zone for block in blocks for zone in scoring_zones(block)[0]}
    return len(WALL_ZONES) - len(occupied)


def construction_score(sites: Iterable[tuple[tuple[int, ...], bool]]) -> int:
    stones = 0
    walls: dict[int, tuple[int, ...]] = {}

    def assign_wall(zones: tuple[int, ...], visited: set[int]) -> bool:
        # Réattribue un mur déjà compté si deux piles se disputent la même zone.
        for zone in zones:
            if zone in visited:
                continue
            visited.add(zone)
            if zone not in walls or assign_wall(walls[zone], visited):
                walls[zone] = zones
                return True
        return False

    for zones, scoring in sites:
        stones += 3 * scoring
        assign_wall(zones, set())
    # TODO : compter les portes et les tours quand le moteur saura représenter ces constructions.
    return stones + 5 * len(walls)


class Simulation:
    def __init__(
        self, obstacles: tuple[Rectangle, ...] = (), *, blocks: tuple[Rectangle, ...] = START_BLOCKS,
        robot: RobotGeometry | None = None,
    ):
        self.obstacles = obstacles
        self._initial_blocks = blocks
        self.geometry = load_robot() if robot is None else robot
        self.reset()

    def reset(self):
        self.time = 0.0
        self.movement: Movement | None = None
        self._pivot: pymunk.PivotJoint | None = None
        self._rolling_target: float | None = None
        self._guard_translation = False
        self._remainder = 0.0
        self._space = pymunk.Space()
        self._space.iterations = 30
        self._space.collision_bias = 0
        self._space.sleep_time_threshold = 0.5
        static = self._space.static_body
        walls = [pymunk.Segment(static, a, b, 0) for a, b in edges(TABLE)]
        obstacles = [pymunk.Poly(static, rectangle_corners(rectangle)) for rectangle in self.obstacles]
        for shape in (*walls, *obstacles):
            shape.friction = 0.7
        self._space.add(*walls, *obstacles)

        self._robot = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._robot.position = START
        robot_shape = pymunk.Poly(self._robot, self.geometry.contour)
        robot_shape.friction = 0.7
        self._space.add(self._robot, robot_shape)

        self._armed_cups: set[int] = set()
        self._piles: list[_Pile] = []
        for left, top, right, bottom in self._initial_blocks:
            width, height = right - left, bottom - top
            size = max(width, height), min(width, height)
            moment = pymunk.moment_for_box(PILE_MASS, size)
            body = pymunk.Body(PILE_MASS, moment)
            body.position = (left + right) / 2, (top + bottom) / 2
            if height > width:
                body.angle = pi / 2
            shape = pymunk.Poly.create_box(body, size)
            shape.friction = 0.6
            # Ces joints freinent le glissement et la rotation sur le tapis, sans ancrer la pile.
            slide = pymunk.PivotJoint(static, body, (0, 0), (0, 0))
            spin = pymunk.GearJoint(static, body, 0, 1)
            slide.max_bias = spin.max_bias = 0
            slide.max_force = PILE_MASS * FLOOR_DECELERATION
            spin.max_force = moment * FLOOR_ANGULAR_DECELERATION
            self._space.add(body, shape, slide, spin)
            self._piles.append(_Pile(body, shape, (slide, spin), size))

    @property
    def heading(self) -> float:
        return self._robot.angle

    @property
    def footprint(self) -> tuple[Point, ...]:
        cosine, sine = cos(self.heading), sin(self.heading)
        return tuple((x * cosine - y * sine, x * sine + y * cosine) for x, y in self.geometry.contour)

    @property
    def moving(self) -> bool:
        return self._robot.body_type == pymunk.Body.DYNAMIC

    @property
    def position(self) -> Point:
        return tuple(self._robot.position)

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(self._block(pile) for pile in self._piles)

    @property
    def score(self) -> int:
        """Points de construction jaunes d'après la pose actuelle des piles libres."""
        return construction_score(scoring_zones(pile.free_block()) for pile in self._piles if pile.grip is None)

    def score_for_blocks(self, blocks: tuple[Block, ...]) -> int:
        """Calcule le score à partir des poses de toutes les piles au pas courant."""
        return construction_score(scoring_zones(block) for pile, block in zip(self._piles, blocks)
                                  if pile.grip is None)

    @property
    def return_score(self) -> int:
        return 0 if self.moving else throne_room_score(self.position, self.footprint)

    def held_pile(self, cup: int) -> int | None:
        return next((index for index, pile in enumerate(self._piles) if pile.grip and pile.grip.cup == cup), None)

    def unload_at_start(self) -> int:
        """Retire les piles portées lorsque le châssis est à l'arrêt dans la zone de départ."""
        if self.return_score != 10:
            return 0
        carried = [pile.grip.shape for pile in self._piles if pile.grip is not None]
        if carried:
            self._space.remove(*carried)
            self._piles = [pile for pile in self._piles if pile.grip is None]
        return len(carried)

    def can_turn_freely(self, angle: float) -> bool:
        polygons = (*(rectangle_corners(obstacle) for obstacle in self.obstacles),
                    *(pile.free_block().corners for pile in self._piles if not pile.grip))
        return self.rotation_clear(self.position, self.footprints(), angle, polygons)

    @classmethod
    def rotation_clear(cls, position, footprints, angle, polygons) -> bool:
        x, y = position
        bounded = [(polygon, _polygon_bounds(polygon)) for polygon in polygons]
        for footprint in footprints:
            destination = _rotated_footprint(footprint, angle)
            if cls._turn_crosses_border(position, footprint, destination, angle):
                return False
            radius = _footprint_radius(footprint) + CONTACT_GAP
            for polygon, (left, top, right, bottom) in bounded:
                if left > x + radius or right < x - radius or top > y + radius or bottom < y - radius:
                    continue
                if cls._turn_hits(position, footprint, destination, angle, polygon):
                    return False
        return True

    @staticmethod
    def _block(pile: _Pile) -> Block:
        grip = pile.grip
        if grip is None:
            return pile.free_block()
        shape = grip.shape
        body = shape.body
        return Block(
            tuple(body.local_to_world(shape.center_of_gravity)),
            body.angle + grip.angle,
            _world_corners(shape),
        )

    @property
    def suction_cups(self) -> tuple[SuctionCup, ...]:
        enabled = self._armed_cups | {pile.grip.cup for pile in self._piles if pile.grip}
        return tuple(
            SuctionCup(tuple(self._robot.local_to_world(mount)), number in enabled)
            for number, (mount, _) in enumerate(self.geometry.mounts, 1)
        )

    def toggle_suction(self, number: int):
        if not isinstance(number, int) or not 1 <= number <= len(self.geometry.mounts):
            raise ValueError("Numéro de ventouse inconnu.")
        if number in self._armed_cups:
            self._armed_cups.remove(number)
            return
        for pile in self._piles:
            grip = pile.grip
            if grip is None or grip.cup != number:
                continue
            center = self._robot.local_to_world(grip.shape.center_of_gravity)
            pile.body.position = center
            pile.body.angle = self.heading + grip.angle
            pile.body.velocity = self._robot.velocity_at_world_point(center)
            pile.body.angular_velocity = self._robot.angular_velocity
            self._space.remove(grip.shape)
            self._space.add(pile.body, pile.shape, *pile.floor)
            pile.grip = None
            return
        self._armed_cups.add(number)
        self._capture_piles()

    def _capture_piles(self):
        captured = False
        for number in sorted(self._armed_cups):
            for pile in self._piles:
                if self._capture(number, pile):
                    captured = True
                    break
        if captured and self.moving and not self.movement.turn:
            previous = self.movement
            self.move_to(*previous.end)
            if previous.blocked:
                self.movement = replace(self.movement, blocked=True)

    def grip_pile(self, number: int, index: int) -> bool:
        """Essaie de saisir la pile désignée avec une ventouse libre, à l'arrêt."""
        return (not self.moving and type(number) is int and 1 <= number <= len(self.geometry.mounts)
                and type(index) is int and 0 <= index < len(self._piles)
                and self.held_pile(number) is None and self._capture(number, self._piles[index]))

    def _capture(self, number: int, pile: _Pile) -> bool:
        if pile.grip:
            return False
        mount, normal = self.geometry.mounts[number - 1]
        point = self._robot.local_to_world(mount)
        x, y = pile.body.world_to_local(point)
        length, width = pile.size
        if abs(x) > length / 2 or abs(abs(y) - width / 2) > SUCTION_REACH:
            return False
        facing = normal.rotated(self.heading - pile.body.angle)
        if (-facing.y if y > 0 else facing.y) < SUCTION_ALIGNMENT:
            return False
        # Une forme sur le châssis garde la pose exacte de la pile pendant les contacts.
        shape = pymunk.Poly(self._robot, [
            self._robot.world_to_local(vertex) for vertex in _world_corners(pile.shape)
        ])
        shape.friction = pile.shape.friction
        self._space.remove(*pile.floor, pile.shape, pile.body)
        self._space.add(shape)
        pile.grip = _Grip(number, shape, pile.body.angle - self.heading)
        self._armed_cups.discard(number)
        return True

    def footprints(self):
        """Contours relatifs au centre du châssis, avec l'orientation actuelle."""
        yield self.footprint
        for pile in self._piles:
            if pile.grip:
                yield tuple(vertex.rotated(self.heading) for vertex in pile.grip.shape.get_vertices())

    def move_to(self, x: float, y: float):
        if not (isfinite(x) and isfinite(y)):
            raise ValueError("La destination doit être finie.")
        start = self.position
        delta = pymunk.Vec2d(x, y) - self._robot.position
        distance = hypot(*delta)
        if not isfinite(distance):
            raise ValueError("La distance à parcourir doit être finie.")
        if self._pivot is not None:
            self._stop()
        self._drive_target = (x, y)
        self._guard_translation = self._rolling_target is not None
        fraction = self._static_travel(start, delta) if distance and not self._guard_translation else 1.0
        blocked = fraction < 1.0
        if blocked:
            fraction = max(0.0, fraction - CONTACT_GAP / distance)
        end = tuple(self._robot.position + delta * fraction) if blocked else (x, y)
        self.movement = Movement(end, blocked)
        self._best_distance = distance * fraction
        self._stall_time = 0.0
        if self._best_distance == 0:
            self._stop()
            return
        if not self.moving:
            self._robot.body_type = pymunk.Body.DYNAMIC
            self._robot.mass = ROBOT_MASS
            self._robot.moment = float("inf")
        direction = delta / distance
        self._robot.velocity = direction * max(0, self._robot.velocity.dot(direction))

    def _static_travel(self, start, delta):
        fraction = 1.0
        for footprint in self.footprints():
            fraction = min(fraction, table_contact(start, delta, footprint))
            for obstacle in self.obstacles:
                fraction = min(fraction, polygon_contact(start, delta, rectangle_corners(obstacle), footprint))
        return fraction

    def _stop(self):
        self._rolling_target = None
        if self._pivot is not None:
            self._space.remove(self._pivot)
            self._pivot = None
            self._robot.angular_velocity = 0
        if self._robot.body_type != pymunk.Body.KINEMATIC:
            self._robot.body_type = pymunk.Body.KINEMATIC
        self._robot.velocity = (0, 0)

    def _step(self):
        if not self.moving:
            self._space.step(PHYSICS_STEP)
            return
        if self._pivot is not None:
            self._step_turn()
            return

        move = self.movement
        delta = pymunk.Vec2d(*move.end) - self._robot.position
        distance = delta.length
        if self._rolling_target is not None:
            angle = remainder(self._rolling_target - self.heading, 2 * pi)
            limit = TURN_SPEED * PHYSICS_STEP
            step_angle = max(-limit, min(limit, angle))
            if self.can_turn_freely(step_angle):
                if abs(angle) <= limit:
                    self._set_heading(self._rolling_target)
                    self._rolling_target = None
                else:
                    self._set_heading(self.heading + step_angle)
            elif distance < 1e-7:
                self.movement = move = replace(move, rotation_blocked=True)
                self._rolling_target = None
        speed = min(SPEED, sqrt(2 * DRIVE_ACCELERATION * distance))
        target_velocity = delta * (min(speed / distance, 1 / PHYSICS_STEP) if distance else 0)
        # Le solveur peut contrer la poussée. La commande retrouve progressivement sa vitesse.
        velocity = self._robot.velocity
        correction = target_velocity - velocity
        max_correction = DRIVE_ACCELERATION * PHYSICS_STEP
        if correction.length > max_correction:
            correction = correction.normalized() * max_correction
        velocity += correction
        if velocity.dot(delta) * PHYSICS_STEP > distance**2:
            velocity = delta / PHYSICS_STEP
        if self._guard_translation and velocity.length > 0:
            travel = velocity * PHYSICS_STEP
            fraction = self._static_travel(self.position, travel)
            if fraction < 1:
                velocity *= max(0, fraction - CONTACT_GAP / travel.length)
        self._robot.velocity = velocity
        self._space.step(PHYSICS_STEP)

        remaining = self._robot.position.get_distance(move.end)
        if remaining < 1e-7:
            self._robot.position = move.end
            if self._rolling_target is None:
                self._stop()
        elif self._stalled(remaining, 0.1):
            self.movement = replace(move, end=self.position, blocked=True)
            self._stop()

    def _stalled(self, remaining: float, progress_threshold: float) -> bool:
        if remaining < self._best_distance - progress_threshold:
            self._best_distance = remaining
            self._stall_time = 0.0
        else:
            self._stall_time += PHYSICS_STEP
        return self._stall_time >= 0.25

    def turn_by(self, angle: float):
        if not -pi < angle < pi:
            raise ValueError("La rotation doit être comprise entre -180° et 180°.")
        if not angle:
            return
        if self.moving and self._pivot is None:
            self._rolling_target = self.heading + angle
            if not self._guard_translation:
                self._guard_translation = True
                self.movement = Movement(self._drive_target, False)
                self._best_distance = self._robot.position.get_distance(self._drive_target)
                self._stall_time = 0.0
            return
        if self._pivot is not None and (self._turn_target - self.heading) * angle > 0:
            return
        position = self.position
        self._stop()
        target = self.heading + angle
        turns = tuple(
            (footprint, tuple(pymunk.Vec2d(*vertex).rotated(angle) for vertex in footprint))
            for footprint in self.footprints()
        )
        fixed = tuple(rectangle_corners(obstacle) for obstacle in self.obstacles)
        blocked = any(
            self._turn_crosses_border(position, footprint, destination, angle)
            or any(self._turn_hits(position, footprint, destination, angle, polygon) for polygon in fixed)
            for footprint, destination in turns
        )
        self.movement = Movement(position, blocked, turn=True)
        if blocked:
            return
        if angle and any(
            self._turn_hits(position, footprint, destination, angle, corners)
            for corners in (pile.free_block().corners for pile in self._piles if not pile.grip)
            for footprint, destination in turns
        ):
            self._robot.body_type = pymunk.Body.DYNAMIC
            self._robot.mass = ROBOT_MASS
            self._robot.moment = pymunk.moment_for_poly(ROBOT_MASS, self.geometry.contour)
            self._pivot = pymunk.PivotJoint(self._space.static_body, self._robot, position)
            self._space.add(self._pivot)
            self._turn_target = target
            self._best_distance = abs(angle)
            self._stall_time = 0.0
            return
        self._set_heading(target)

    def _step_turn(self):
        delta = self._turn_target - self.heading
        target_speed = max(-TURN_SPEED, min(TURN_SPEED, delta / PHYSICS_STEP))
        correction = target_speed - self._robot.angular_velocity
        max_correction = TURN_ACCELERATION * PHYSICS_STEP
        if abs(delta) > TURN_SPEED * PHYSICS_STEP:
            correction = max(-max_correction, min(max_correction, correction))
        self._robot.angular_velocity += correction
        self._space.step(PHYSICS_STEP)
        remaining = abs(self._turn_target - self.heading)
        if remaining < 1e-7:
            self._stop()
            self._set_heading(self._turn_target)
        elif self._stalled(remaining, radians(0.05)):
            self.movement = replace(self.movement, end=self.position, blocked=True)
            self._stop()
            self._set_heading(self.heading)

    def _set_heading(self, heading: float):
        self._robot.angle = (heading + pi) % (2 * pi) - pi
        self._space.reindex_shapes_for_body(self._robot)

    @staticmethod
    def _turn_crosses_border(position: Point, footprint: tuple[Point, ...],
                             destination: tuple[Point, ...], angle: float) -> bool:
        x, y = position
        if any(not (0 <= x + dx <= TABLE_WIDTH and 0 <= y + dy <= TABLE_HEIGHT) for dx, dy in destination):
            return True
        radius = _footprint_radius(footprint)
        if min(x, y, TABLE_WIDTH - x, TABLE_HEIGHT - y) < radius:
            border = tuple((px - x, py - y) for px, py in TABLE)
            if any(arc_hits_edge(vertex, angle, a, b) for vertex in footprint for a, b in edges(border)):
                return True
        return False

    @staticmethod
    def _turn_hits(position: Point, footprint: tuple[Point, ...], destination: tuple[Point, ...],
                   angle: float, polygon: tuple[Point, ...]) -> bool:
        if polygon_contact(position, (0, 0), polygon, destination) < 1:
            return True
        x, y = position
        corners = tuple((px - x, py - y) for px, py in polygon)
        if any(arc_hits_edge(vertex, angle, a, b) for vertex in footprint for a, b in edges(corners)):
            return True
        # Dans le repère du robot, la pile tourne dans l'autre sens.
        return any(arc_hits_edge(vertex, -angle, a, b) for vertex in corners for a, b in edges(footprint))

    @property
    def settled(self) -> bool:
        return (not self.moving and not self._armed_cups
                and all(pile.grip or pile.body.is_sleeping for pile in self._piles))

    def advance(self, seconds: float):
        if not isfinite(seconds) or seconds < 0:
            raise ValueError("La durée doit être finie et positive ou nulle.")
        self.time += seconds
        self._remainder += seconds
        steps = int((self._remainder + 1e-12) / PHYSICS_STEP)
        self._remainder = max(0.0, self._remainder - steps * PHYSICS_STEP)
        for _ in range(steps):
            if self.settled:
                break
            self._step()
            if self._armed_cups:
                self._capture_piles()
