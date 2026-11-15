"""Contrat de sélection des candidats."""

from typing import Protocol


class Selecteur(Protocol):
    id: str

    def classer(self, observation, candidats): ...

    def observer(self, candidat, valeur: float) -> None: ...


def choisir(selecteur: Selecteur, observation, candidats):
    ordonnes = selecteur.classer(observation, candidats)
    return ordonnes[0] if ordonnes else None
