"""Chemins des ressources du projet, injectables pour les tests."""

from dataclasses import dataclass
from pathlib import Path


RACINE = Path(__file__).resolve().parents[2]
REPLAY_DIR = RACINE / "replays"
MODELE = RACINE / "reseau.npz"
TAPIS = RACINE / "assets" / "tapis_2027.png"
ROBOT_FILE = RACINE / "RobotRobotronik.toml"


@dataclass(frozen=True)
class Ressources:
    racine: Path = RACINE
    replays: Path = REPLAY_DIR
    modele: Path = MODELE
    tapis: Path = TAPIS
    robot: Path = ROBOT_FILE
