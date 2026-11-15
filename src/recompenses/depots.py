"""Récompense par dépose réussie."""

from .base import Transition


class Depots:
    id = "depots"

    def valeur(self, transition: Transition) -> float:
        return 1.0 if transition.candidat.action == "depose" else 0.0

    def rang(self, replay) -> tuple[float, float]:
        return replay.get("depots", 0), -replay["duree"]
