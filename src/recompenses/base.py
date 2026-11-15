"""Contrat de récompense."""

from dataclasses import dataclass
from typing import Mapping, Protocol

from ..domaine.modele import Candidat, Observation


@dataclass(frozen=True)
class Transition:
    avant: Observation
    candidat: Candidat
    apres: Observation
    replay: Mapping | None = None


class Recompense(Protocol):
    id: str

    def valeur(self, transition: Transition) -> float: ...

    def rang(self, replay: Mapping) -> tuple[float, float]: ...
