"""Profil de composition."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

from ..infra.ressources import MODELE


@dataclass(frozen=True)
class Profil:
    robot: str = "robotronik"
    terrain: str = "officiel_2027"
    scenario: str = "officiel_2027"
    planification: str = "astar"
    recompense: str = "construction_temps"
    selection: str = "glouton"
    reseau: str = "mlp"
    modele: Path = MODELE


def defaut() -> Profil:
    return Profil()


def charger(path: Path) -> Profil:
    with Path(path).open("rb") as file:
        data = tomllib.load(file)

    def valeur(section: str, cle: str, defaut_valeur):
        return data.get(section, {}).get(cle, defaut_valeur)

    return Profil(
        robot=valeur("robot", "id", "robotronik"),
        terrain=valeur("terrain", "id", "officiel_2027"),
        scenario=valeur("scenario", "id", "officiel_2027"),
        planification=valeur("planification", "id", "astar"),
        recompense=valeur("recompense", "id", "construction_temps"),
        selection=valeur("selection", "id", "glouton"),
        reseau=valeur("reseau", "id", "mlp"),
        modele=Path(valeur("reseau", "modele", str(MODELE))),
    )
