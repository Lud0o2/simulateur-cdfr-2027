"""Création d'une pile physique à partir de son rectangle initial."""

from math import pi

import pymunk

from .frottement import ajouter_freinage_sol


def creer_pile(
    space, static, rectangle, *, masse, deceleration, angular_deceleration, frottement=0.6,
):
    left, top, right, bottom = rectangle
    width, height = right - left, bottom - top
    size = max(width, height), min(width, height)
    moment = pymunk.moment_for_box(masse, size)
    body = pymunk.Body(masse, moment)
    body.position = (left + right) / 2, (top + bottom) / 2
    if height > width:
        body.angle = pi / 2
    shape = pymunk.Poly.create_box(body, size)
    shape.friction = frottement
    floor = ajouter_freinage_sol(
        space, static, body, moment, masse=masse, deceleration=deceleration,
        angular_deceleration=angular_deceleration,
    )
    space.add(body, shape, *floor)
    return body, shape, floor, size
