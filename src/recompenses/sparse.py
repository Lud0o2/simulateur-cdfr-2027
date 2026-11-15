"""Récompense uniquement en fin d'épisode."""

from .base import Transition


class Sparse:
    id = "sparse"

    def valeur(self, transition: Transition) -> float:
        return 0.0

    def rang(self, replay) -> tuple[float, float]:
        return replay["score"], -replay["duree"]
