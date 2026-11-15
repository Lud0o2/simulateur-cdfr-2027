"""Composition par défaut : comportement actuel du simulateur."""

from dataclasses import dataclass
from pathlib import Path

from ..actions import ACTIONS
from ..domaine.scenario import ScenarioSpec
from ..evaluation.scenarios import officiel_2027
from ..planification.astar import AstarPlanner
from ..recompenses.construction_temps import ConstructionTemps
from ..reseaux.caracteristiques import EncodeurCaracteristiques
from .registry import resoudre
from ..selection.glouton import Glouton
from ..selection.reseau import MatchPolicy
from .config import Profil, defaut


@dataclass(frozen=True)
class Composition:
    scenario: ScenarioSpec
    planificateur: object
    actions: tuple
    recompense: object
    selecteur: object
    reseau: object | None
    encodeur: object | None


def composer(profil: Profil | None = None) -> Composition:
    profil = defaut() if profil is None else profil
    scenario = officiel_2027()
    reseau = None
    selecteur = Glouton()
    encodeur = None
    if profil.selection == "reseau" and Path(profil.modele).exists():
        reseau = resoudre("reseau", profil.reseau).load(profil.modele)
        selecteur = MatchPolicy(reseau)
        encodeur = EncodeurCaracteristiques()
    actions = tuple(
        action for action in ACTIONS
        if action.construction is None or scenario.robot.supports(action.construction)
    )
    return Composition(
        scenario=scenario,
        planificateur=AstarPlanner(),
        actions=actions,
        recompense=ConstructionTemps(),
        selecteur=selecteur,
        reseau=reseau,
        encodeur=encodeur,
    )
