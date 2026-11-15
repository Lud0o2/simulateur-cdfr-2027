"""Vue géométrique compatible avec le moteur Pymunk."""

from dataclasses import dataclass

import pymunk

from ..domaine.modele import Point
from ..domaine.robot import Construction, Mouvement, RobotSpec, Ventouse


@dataclass(frozen=True, slots=True)
class RobotGeometry:
    contour: tuple[Point, ...]
    mounts: tuple[tuple[Point, pymunk.Vec2d], ...]


def geometry_from_spec(spec: RobotSpec) -> RobotGeometry:
    return RobotGeometry(
        spec.contour,
        tuple((ventouse.position_mm, pymunk.Vec2d(*ventouse.face)) for ventouse in spec.ventouses),
    )


def spec_from_geometry(
    geometry: RobotGeometry, *, identifiant: str = "robot",
    mouvement: Mouvement | None = None, planificateur: str = "astar",
) -> RobotSpec:
    ventouses = tuple(
        Ventouse(number, mount, (facing.x, facing.y), Construction.MUR)
        for number, (mount, facing) in enumerate(geometry.mounts, 1)
    )
    return RobotSpec(identifiant, geometry.contour, ventouses, mouvement or Mouvement(), planificateur)
