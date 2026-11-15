"""Récompense fondée sur le score de construction."""

from .base import Transition


class Construction:
    id = "construction"

    def valeur(self, transition: Transition) -> float:
        return (transition.apres.score - transition.avant.score) / 8

    def rang(self, replay) -> tuple[float, float]:
        return replay["score"], -replay["duree"]
