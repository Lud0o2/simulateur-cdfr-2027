"""Description d'un robot et de ses capacités."""

from dataclasses import dataclass, field
from enum import StrEnum
from math import radians
from typing import Protocol

from .modele import Point


class Construction(StrEnum):
    MUR = "mur"
    TOUR = "tour"
    PORTE = "porte"


CAPACITES = {
    Construction.MUR: frozenset({Construction.MUR}),
    Construction.TOUR: frozenset({Construction.TOUR, Construction.MUR}),
    Construction.PORTE: frozenset({Construction.PORTE, Construction.TOUR, Construction.MUR}),
}


@dataclass(frozen=True)
class Ventouse:
    numero: int
    position_mm: Point
    face: Point
    construction: Construction = Construction.MUR

    @property
    def capacites(self) -> frozenset[Construction]:
        return CAPACITES[self.construction]


@dataclass(frozen=True)
class Mouvement:
    type: str = "holonome"
    vitesse_mm_s: float = 1000.0
    acceleration_mm_s2: float = 1000.0
    vitesse_rotation_rad_s: float = radians(360)
    acceleration_rotation_rad_s2: float = radians(1200)
    pas_rotation_rad: float = radians(15)
    rotation_sur_place: bool = True
    deplacement_lateral: bool = True
    masse_kg: float = 5.0


@dataclass(frozen=True)
class RobotSpec:
    id: str
    contour: tuple[Point, ...]
    ventouses: tuple[Ventouse, ...] = ()
    mouvement: Mouvement = field(default_factory=Mouvement)
    planificateur: str = "astar"

    def supports(self, construction: Construction) -> bool:
        return any(construction in ventouse.capacites for ventouse in self.ventouses)


class Prehenseur(Protocol):
    """Contrat minimal d'un effecteur de prise."""

    id: str

    def peut_prendre(self, simulation, numero: int, index: int) -> bool: ...

    def prendre(self, simulation, numero: int, index: int) -> bool: ...

    def relacher(self, simulation, numero: int) -> bool: ...
