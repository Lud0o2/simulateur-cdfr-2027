"""Modèles de coût interchangeables pour la planification."""

from typing import Protocol

from ..domaine.geometrie import travel_time


class ModeleCout(Protocol):
    id: str

    def cout(self, distance: float) -> float: ...


class CoutTemps:
    id = "temps"

    def __init__(self, vitesse: float = 1000.0, acceleration: float = 1000.0):
        self.vitesse = vitesse
        self.acceleration = acceleration

    def cout(self, distance: float) -> float:
        return travel_time(distance, self.vitesse, self.acceleration)


class CoutDistance:
    id = "distance"

    def cout(self, distance: float) -> float:
        return distance
