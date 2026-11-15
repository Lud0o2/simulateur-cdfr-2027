"""Récompense qui inclut le bonus de salle du trône."""

from .base import Transition


class Retour:
    id = "retour"

    def valeur(self, transition: Transition) -> float:
        replay = transition.replay or {}
        retour = replay.get("score_retour", 0)
        return (transition.apres.score - transition.avant.score + retour) / 8

    def rang(self, replay) -> tuple[float, float]:
        return replay.get("score", 0), -replay["duree"]
