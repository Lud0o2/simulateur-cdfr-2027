"""Récompense actuelle : score final et bonus de temps."""

from ..regles.phases import MATCH_SECONDS
from .base import Transition


class ConstructionTemps:
    id = "construction_temps"

    def valeur(self, transition: Transition) -> float:
        replay = transition.replay or {}
        duree = replay.get("duree", MATCH_SECONDS)
        score = replay.get("score", transition.apres.score)
        bonus = 0.5 * (1 - duree / MATCH_SECONDS)
        return (score - transition.avant.score + bonus) / 8

    def rang(self, replay) -> tuple[float, float]:
        return replay["score"], -replay["duree"]
