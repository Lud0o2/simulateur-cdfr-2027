"""Récompense actuelle : score final et bonus de temps."""

from ..regles.phases import MATCH_SECONDS
from .base import Transition


class ConstructionTemps:
    id = "construction_temps"

    def valeur_match(self, score_avant: float, replay) -> float:
        duree = replay.get("duree", MATCH_SECONDS)
        bonus = 0.5 * (1 - duree / MATCH_SECONDS)
        return (replay["score"] - score_avant + bonus) / 8

    def valeur(self, transition: Transition) -> float:
        replay = transition.replay or {}
        return self.valeur_match(transition.avant.score, replay)

    def rang(self, replay) -> tuple[float, float]:
        return replay["score"], -replay["duree"]
