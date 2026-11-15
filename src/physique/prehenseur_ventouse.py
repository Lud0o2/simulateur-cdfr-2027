"""Capture et relâchement d'une pile par une ventouse."""

import pymunk

from .collisions import _world_corners
from .constantes import SUCTION_ALIGNMENT, SUCTION_REACH
from .corps import _Grip


def relacher(simulation, number: int) -> bool:
    for pile in simulation._piles:
        grip = pile.grip
        if grip is None or grip.cup != number:
            continue
        center = simulation._robot.local_to_world(grip.shape.center_of_gravity)
        pile.body.position = center
        pile.body.angle = simulation.heading + grip.angle
        pile.body.velocity = simulation._robot.velocity_at_world_point(center)
        pile.body.angular_velocity = simulation._robot.angular_velocity
        simulation._space.remove(grip.shape)
        simulation._space.add(pile.body, pile.shape, *pile.floor)
        pile.grip = None
        return True
    return False


def capturer(simulation, number: int, pile) -> bool:
    if pile.grip:
        return False
    mount, normal = simulation.geometry.mounts[number - 1]
    point = simulation._robot.local_to_world(mount)
    x, y = pile.body.world_to_local(point)
    length, width = pile.size
    if abs(x) > length / 2 or abs(abs(y) - width / 2) > SUCTION_REACH:
        return False
    facing = normal.rotated(simulation.heading - pile.body.angle)
    if (-facing.y if y > 0 else facing.y) < SUCTION_ALIGNMENT:
        return False
    # Une forme sur le châssis garde la pose exacte de la pile pendant les contacts.
    shape = pymunk.Poly(simulation._robot, [
        simulation._robot.world_to_local(vertex) for vertex in _world_corners(pile.shape)
    ])
    shape.friction = pile.shape.friction
    simulation._space.remove(*pile.floor, pile.shape, pile.body)
    simulation._space.add(shape)
    pile.grip = _Grip(number, shape, pile.body.angle - simulation.heading)
    simulation._armed_cups.discard(number)
    return True
