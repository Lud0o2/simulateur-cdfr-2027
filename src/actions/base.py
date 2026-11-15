"""Contrats d'action."""

from dataclasses import dataclass
from typing import Any, Protocol

from ..domaine.modele import Candidat


@dataclass(frozen=True)
class ContexteAction:
    moteur: Any
    robot: Any
    planificateur: Any
    terrain: Any
    temps: float
    deadline: float


class Action(Protocol):
    id: str

    def candidats(self, ctx: ContexteAction): ...

    def executer(self, ctx: ContexteAction, candidat: Candidat) -> bool: ...
