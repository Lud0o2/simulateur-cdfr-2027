"""Contrats et données de planification."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

from pymunk import Vec2d

from ..domaine.geometrie import travel_time


@dataclass(frozen=True)
class Approach:
    duration: float
    before: tuple[Vec2d, ...]
    turn: float
    after: tuple[Vec2d, ...]
    rolling: bool = False

    def distance_from(self, start):
        return sum(a.get_distance(b) for a, b in pairwise((start, *self.before, *self.after)))


@dataclass(frozen=True)
class Pickup:
    approach: Approach
    end: Vec2d

    @property
    def duration(self):
        return self.approach.duration + travel_time(self.end.get_distance((self.approach.after or self.approach.before)[-1]))

    def distance_from(self, start):
        contact = (self.approach.after or self.approach.before)[-1]
        return self.approach.distance_from(start) + contact.get_distance(self.end)


@dataclass(frozen=True)
class Plan:
    pile: int
    cup: int
    pickup: Pickup
    drop: tuple[Vec2d, ...]
    retreat: tuple[Vec2d, ...]




@dataclass(frozen=True)
class DemandeTrajet:
    depart: tuple[float, float]
    arrivee: tuple[float, float]
    empreintes: tuple[tuple[tuple[float, float], ...], ...]
    obstacles: tuple[tuple[tuple[float, float], ...], ...]
    marge: float = 5.0
    angle_vise: float = 0.0
    rotation_en_roulant: bool = True
    limite_s: float = float("inf")


@dataclass(frozen=True)
class Trajet:
    duree_s: float
    avant: tuple[tuple[float, float], ...]
    rotation_rad: float
    apres: tuple[tuple[float, float], ...]
    en_roulant: bool = False


class Planificateur(Protocol):
    id: str

    def planifier(self, demande: DemandeTrajet) -> Trajet | None: ...
