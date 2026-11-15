"""Adaptateurs Pymunk pour les requêtes de collision."""

import pymunk

from ..domaine.modele import Point


def _world_corners(shape: pymunk.Poly) -> tuple[Point, ...]:
    return tuple(tuple(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices())
