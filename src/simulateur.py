from dataclasses import dataclass, replace
from functools import cache
from math import atan2, cos, hypot, isfinite, pi, radians, sin, sqrt
from pathlib import Path
import tomllib

import pymunk


Point = tuple[float, float]
Rectangle = tuple[float, float, float, float]
TABLE_WIDTH = 3000
TABLE_HEIGHT = 2000
SPEED = 1000  # mm/s
TURN_STEP = radians(15)
TURN_SPEED = radians(180)
TURN_ACCELERATION = radians(1200)  # rad/s², effort de pivot provisoire
ROBOT_FILE = Path(__file__).resolve().parents[1] / "RobotRobotronik.toml"
START = (250.0, 1000.0)
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
DRIVE_ACCELERATION = 6000  # mm/s², effort disponible pendant la poussée
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


def _world_corners(shape: pymunk.Poly) -> tuple[Point, ...]:
    return tuple(tuple(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices())


def polygon_contact(
    position: Point, delta: Point, corners: tuple[Point, ...], footprint: tuple[Point, ...]
) -> float:
    """Renvoie la fraction du trajet où les deux polygones commencent à se chevaucher."""
    x, y = position
    dx, dy = delta
    enter, leave = 0.0, 1.0
    for (ax, ay), (bx, by) in (*edges(footprint), *edges(corners)):
        ax, ay = by - ay, ax - bx
        center = x * ax + y * ay
        offsets = [px * ax + py * ay for px, py in footprint]
        low, high = center + min(offsets), center + max(offsets)
        bounds = [px * ax + py * ay for px, py in corners]
        other_low, other_high = min(bounds), max(bounds)
        speed = dx * ax + dy * ay
        if speed == 0:
            if high <= other_low or low >= other_high:
                return 1.0
            continue
        first = (other_low - high) / speed
        last = (other_high - low) / speed
        enter = max(enter, min(first, last))
        leave = min(leave, max(first, last))
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


@dataclass(frozen=True)
class Movement:
    end: Point
    blocked: bool
    turn: bool = False


@dataclass(frozen=True)
class Block:
    position: Point
    angle: float
    corners: tuple[Point, ...]


@dataclass(frozen=True)
class SuctionCup:
    position: Point
    enabled: bool
    holding: bool = False


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


class Simulation:
    def __init__(
        self, obstacles: tuple[Rectangle, ...] = (), *, blocks: tuple[Rectangle, ...] = START_BLOCKS,
        robot: RobotGeometry | None = None,
    ):
        self.obstacles = obstacles
        self._initial_blocks = blocks
        self._geometry = load_robot() if robot is None else robot
        self.reset()

    def reset(self):
        self.time = 0.0
        self.movement: Movement | None = None
        self._pivot: pymunk.PivotJoint | None = None
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
        robot_shape = pymunk.Poly(self._robot, self._geometry.contour)
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
        return tuple((x * cosine - y * sine, x * sine + y * cosine) for x, y in self._geometry.contour)

    @property
    def moving(self) -> bool:
        return self._robot.body_type == pymunk.Body.DYNAMIC

    @property
    def position(self) -> Point:
        return tuple(self._robot.position)

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(self._block(pile) for pile in self._piles)

    @staticmethod
    def _block(pile: _Pile) -> Block:
        grip = pile.grip
        shape = grip.shape if grip else pile.shape
        body = shape.body
        return Block(
            tuple(body.local_to_world(shape.center_of_gravity)),
            body.angle + grip.angle if grip else body.angle,
            _world_corners(shape),
        )

    @property
    def suction_cups(self) -> tuple[SuctionCup, ...]:
        holding = {pile.grip.cup for pile in self._piles if pile.grip}
        enabled = self._armed_cups | holding
        return tuple(
            SuctionCup(tuple(self._robot.local_to_world(mount)), number in enabled, number in holding)
            for number, (mount, _) in enumerate(self._geometry.mounts, 1)
        )

    def toggle_suction(self, number: int):
        if not isinstance(number, int) or not 1 <= number <= len(self._geometry.mounts):
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
            mount, normal = self._geometry.mounts[number - 1]
            point = self._robot.local_to_world(mount)
            for pile in self._piles:
                if pile.grip:
                    continue
                x, y = pile.body.world_to_local(point)
                length, width = pile.size
                if abs(x) > length / 2 or abs(abs(y) - width / 2) > SUCTION_REACH:
                    continue
                facing = normal.rotated(self.heading - pile.body.angle)
                if (-facing.y if y > 0 else facing.y) < SUCTION_ALIGNMENT:
                    continue
                # Une forme sur le châssis garde la pose exacte de la pile pendant les contacts.
                shape = pymunk.Poly(self._robot, [
                    self._robot.world_to_local(vertex) for vertex in _world_corners(pile.shape)
                ])
                shape.friction = pile.shape.friction
                self._space.remove(*pile.floor, pile.shape, pile.body)
                self._space.add(shape)
                pile.grip = _Grip(number, shape, pile.body.angle - self.heading)
                self._armed_cups.remove(number)
                captured = True
                break
        if captured and self.moving and not self.movement.turn:
            previous = self.movement
            self.move_to(*previous.end)
            if previous.blocked:
                self.movement = replace(self.movement, blocked=True)

    def _footprints(self):
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
        fraction = 1.0
        if distance > 0:
            for footprint in self._footprints():
                fraction = min(fraction, table_contact(start, delta, footprint))
                for obstacle in self.obstacles:
                    fraction = min(fraction, polygon_contact(start, delta, rectangle_corners(obstacle), footprint))
        blocked = fraction < 1.0
        if blocked:
            fraction = max(0.0, fraction - CONTACT_GAP / distance)
        end = tuple(self._robot.position + delta * fraction) if blocked else (x, y)
        previous = self.movement
        self.movement = Movement(end, blocked)
        self._best_distance = distance * fraction
        self._stall_time = 0.0
        if self._best_distance == 0:
            self._stop()
            return
        was_moving = self.moving
        if not was_moving:
            self._robot.body_type = pymunk.Body.DYNAMIC
            self._robot.mass = ROBOT_MASS
            self._robot.moment = float("inf")
        direction = delta / distance
        new_speed = min(SPEED, self._best_distance / PHYSICS_STEP)
        if not was_moving or self._robot.velocity.length >= SPEED - 1e-6:
            self._robot.velocity = direction * new_speed
        elif (pymunk.Vec2d(*previous.end) - self._robot.position).dot(direction) < 0:
            # En repartant dans l'autre sens, le robot holonome quitte le contact.
            self._robot.velocity = direction * min(new_speed, self._robot.velocity.length)

    def _stop(self):
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
        target_velocity = delta / PHYSICS_STEP
        if distance > SPEED * PHYSICS_STEP:
            target_velocity = delta * (SPEED / distance)
        # Le solveur peut contrer la poussée. La commande retrouve progressivement sa vitesse.
        correction = target_velocity - self._robot.velocity
        if correction.length > DRIVE_ACCELERATION * PHYSICS_STEP and distance > SPEED * PHYSICS_STEP:
            correction = correction.normalized() * DRIVE_ACCELERATION * PHYSICS_STEP
        self._robot.velocity += correction
        self._space.step(PHYSICS_STEP)

        remaining = self._robot.position.get_distance(move.end)
        if remaining < 1e-7:
            self._robot.position = move.end
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
        if self._pivot is not None and (self._turn_target - self.heading) * angle > 0:
            return
        position = self.position
        self._stop()
        target = self.heading + angle
        turns = tuple(
            (footprint, tuple(pymunk.Vec2d(*vertex).rotated(angle) for vertex in footprint))
            for footprint in self._footprints()
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
            for corners in (_world_corners(pile.shape) for pile in self._piles if not pile.grip)
            for footprint, destination in turns
        ):
            self._robot.body_type = pymunk.Body.DYNAMIC
            self._robot.mass = ROBOT_MASS
            self._robot.moment = pymunk.moment_for_poly(ROBOT_MASS, self._geometry.contour)
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

    def _turn_crosses_border(self, position: Point, footprint: tuple[Point, ...],
                             destination: tuple[Point, ...], angle: float) -> bool:
        x, y = position
        if any(not (0 <= x + dx <= TABLE_WIDTH and 0 <= y + dy <= TABLE_HEIGHT) for dx, dy in destination):
            return True
        radius = max(hypot(*vertex) for vertex in footprint)
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

    def advance(self, seconds: float):
        if not isfinite(seconds) or seconds < 0:
            raise ValueError("La durée doit être finie et positive ou nulle.")
        self.time += seconds
        self._remainder += seconds
        steps = int((self._remainder + 1e-12) / PHYSICS_STEP)
        self._remainder = max(0.0, self._remainder - steps * PHYSICS_STEP)
        for _ in range(steps):
            self._step()
            if self._armed_cups:
                self._capture_piles()
