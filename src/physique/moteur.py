"""Moteur physique Pymunk et état d'un match."""

from dataclasses import replace
from math import cos, hypot, isfinite, pi, radians, remainder, sin, sqrt
import pymunk

from ..domaine.constantes import CONTACT_GAP
from ..domaine.geometrie import (
    _footprint_radius, _polygon_bounds, _rotated_vertices, arc_hits_edge, edges,
    polygon_contact, rectangle_corners, table_contact as _table_contact,
    throne_room_score as _throne_room_score,
)
from ..domaine.modele import Block, Movement, Point, SuctionCup
from ..regles.score_2027 import construction_score, scoring_zones, table_contact, throne_room_score
from ..robots.spec import RobotGeometry
from ..terrains.officiel_2027 import START_BLOCKS, TERRAIN
from .collisions import _world_corners
from .constantes import (
    DRIVE_ACCELERATION, FLOOR_ANGULAR_DECELERATION, FLOOR_DECELERATION, PHYSICS_STEP,
    PILE_MASS, ROBOT_MASS, SPEED, SUCTION_ALIGNMENT, SUCTION_REACH, TURN_ACCELERATION,
    TURN_SPEED,
)
from .corps import _Grip, _Pile
from .pile import creer_pile
from .prehenseur_ventouse import capturer, relacher


class Simulation:
    def __init__(
        self, obstacles: tuple[Rectangle, ...] = (), *, blocks: tuple[Rectangle, ...] | None = None,
        robot: RobotGeometry | None = None, terrain=None,
    ):
        self.terrain = TERRAIN if terrain is None else terrain
        self.obstacles = obstacles
        self._initial_blocks = self.terrain.initial_blocks if blocks is None else blocks
        if robot is None:
            from ..robots.loader import load_robot
            robot = load_robot()
        self.geometry = robot
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
        width, height = self.terrain.width, self.terrain.height
        table = ((0, 0), (width, 0), (width, height), (0, height))
        walls = [pymunk.Segment(static, a, b, 0) for a, b in edges(table)]
        obstacles = [pymunk.Poly(static, rectangle_corners(rectangle)) for rectangle in self.obstacles]
        for shape in (*walls, *obstacles):
            shape.friction = 0.7
        self._space.add(*walls, *obstacles)

        self._robot = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._robot.position = self.terrain.start[:2]
        robot_shape = pymunk.Poly(self._robot, self.geometry.contour)
        robot_shape.friction = 0.7
        self._space.add(self._robot, robot_shape)

        self._armed_cups: set[int] = set()
        self._piles: list[_Pile] = []
        for rectangle in self._initial_blocks:
            body, shape, floor, size = creer_pile(
                self._space, static, rectangle, masse=PILE_MASS,
                deceleration=FLOOR_DECELERATION, angular_deceleration=FLOOR_ANGULAR_DECELERATION,
            )
            self._piles.append(_Pile(body, shape, floor, size))

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
        return self.score_for_blocks(self.blocks)

    def score_for_blocks(self, blocks: tuple[Block, ...]) -> int:
        """Calcule le score à partir des poses de toutes les piles au pas courant."""
        return construction_score(scoring_zones(block) for pile, block in zip(self._piles, blocks)
                                  if pile.grip is None)

    @property
    def return_score(self) -> int:
        return 0 if self.moving else _throne_room_score(self.position, self.footprint, self.terrain.throne_room)

    def held_pile(self, cup: int) -> int | None:
        return next((index for index, pile in enumerate(self._piles) if pile.grip and pile.grip.cup == cup), None)

    def can_turn_freely(self, angle: float) -> bool:
        polygons = (*(rectangle_corners(obstacle) for obstacle in self.obstacles),
                    *(pile.free_block().corners for pile in self._piles if not pile.grip))
        return self.rotation_clear(self.position, self.footprints(), angle, polygons)

    @classmethod
    def rotation_clear(cls, position, footprints, angle, polygons) -> bool:
        x, y = position
        bounded = [(polygon, _polygon_bounds(polygon)) for polygon in polygons]
        for footprint in footprints:
            destination = _rotated_vertices(footprint, angle)
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
        if relacher(self, number):
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
        return capturer(self, number, pile)

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
            fraction = min(fraction, _table_contact(
                start, delta, footprint, width=self.terrain.width, height=self.terrain.height,
            ))
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
            self._turn_crosses_border(
                position, footprint, destination, angle,
                width=self.terrain.width, height=self.terrain.height,
            )
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
    def _turn_crosses_border(
        position: Point, footprint: tuple[Point, ...], destination: tuple[Point, ...], angle: float,
        *, width: float = TERRAIN.width, height: float = TERRAIN.height,
    ) -> bool:
        x, y = position
        if any(not (0 <= x + dx <= width and 0 <= y + dy <= height) for dx, dy in destination):
            return True
        radius = _footprint_radius(footprint)
        if min(x, y, width - x, height - y) < radius:
            table = ((0, 0), (width, 0), (width, height), (0, height))
            border = tuple((px - x, py - y) for px, py in table)
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
            if not self.moving and not self._armed_cups and all(pile.grip or pile.body.is_sleeping for pile in self._piles):
                break
            self._step()
            if self._armed_cups:
                self._capture_piles()
