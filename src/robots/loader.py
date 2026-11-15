"""Chargement et validation du fichier TOML du robot."""

from functools import cache
from math import isfinite, radians
from pathlib import Path
import tomllib

import pymunk

from ..domaine.geometrie import edges
from ..domaine.robot import Construction, Mouvement, RobotSpec, Ventouse
from ..domaine.validation import _finite_number, _read_point
from ..terrains.officiel_2027 import START, TABLE_HEIGHT, TABLE_WIDTH
from .spec import RobotGeometry, geometry_from_spec


ROBOT_FILE = Path(__file__).resolve().parents[2] / "RobotRobotronik.toml"
_ROBOT_KEYS = {"nom", "contour_mm", "ventouses", "mouvement", "planificateur"}
_CUP_KEYS = {"position_mm", "angle_deg", "construction"}
_MOUVEMENT_KEYS = {
    "type", "vitesse_mm_s", "acceleration_mm_s2", "vitesse_rotation_rad_s",
    "acceleration_rotation_rad_s2", "pas_rotation_rad", "rotation_sur_place",
    "deplacement_lateral", "masse_kg",
}


def _read_mouvement(data) -> Mouvement:
    if data is None:
        return Mouvement()
    if not isinstance(data, dict):
        raise ValueError("mouvement doit être une table TOML.")
    unknown = data.keys() - _MOUVEMENT_KEYS
    if unknown:
        raise ValueError(f"Champs de mouvement inconnus : {', '.join(sorted(unknown))}.")
    values = {key: data[key] for key in _MOUVEMENT_KEYS if key in data}
    for key in ("vitesse_mm_s", "acceleration_mm_s2", "vitesse_rotation_rad_s",
                "acceleration_rotation_rad_s2", "pas_rotation_rad", "masse_kg"):
        if key in values and not (_finite_number(values[key]) and values[key] > 0):
            raise ValueError(f"mouvement.{key} doit être un nombre positif fini.")
    for key in ("rotation_sur_place", "deplacement_lateral"):
        if key in values and type(values[key]) is not bool:
            raise ValueError(f"mouvement.{key} doit être un booléen.")
    if "type" in values and values["type"] not in {"holonome", "differentiel"}:
        raise ValueError("mouvement.type doit valoir holonome ou differentiel.")
    return Mouvement(**values)


@cache
def load_robot_spec(path: Path = ROBOT_FILE) -> RobotSpec:
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
        unknown = data.keys() - _ROBOT_KEYS
        if unknown:
            raise ValueError(f"Les champs autorisés sont {', '.join(sorted(_ROBOT_KEYS))}.")
        points = data.get("contour_mm")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("contour_mm doit contenir au moins trois sommets.")
        contour = tuple(_read_point(point, f"contour_mm, sommet {index}")
                        for index, point in enumerate(points, 1))
        if any(not (0 <= START[0] + x <= TABLE_WIDTH and 0 <= START[1] + y <= TABLE_HEIGHT)
               for x, y in contour):
            raise ValueError("Le contour dépasse de la table à la position de départ.")
        vertices = tuple(pymunk.Vec2d(*point) for point in contour)
        turns = [(b - a).cross(point - a) for a, b in edges(vertices)
                 for point in vertices if point != a and point != b]
        if len(set(contour)) != len(contour) or not (
            all(isfinite(turn) and turn > 0 for turn in turns)
            or all(isfinite(turn) and turn < 0 for turn in turns)
        ):
            raise ValueError("contour_mm doit décrire un contour convexe, sans sommets alignés, dans l’ordre du périmètre.")

        cups = data.get("ventouses", [])
        if not isinstance(cups, list):
            raise ValueError("Déclare les ventouses avec des blocs [[ventouses]].")
        ventouses = []
        for number, cup in enumerate(cups, 1):
            if not isinstance(cup, dict) or not {"position_mm", "angle_deg"} <= cup.keys() <= _CUP_KEYS:
                raise ValueError(f"La ventouse {number} doit définir position_mm et angle_deg.")
            mount = _read_point(cup["position_mm"], f"Ventouse {number}, position_mm")
            angle = cup["angle_deg"]
            if not _finite_number(angle):
                raise ValueError(f"Ventouse {number}, angle_deg doit être un angle fini en degrés.")
            try:
                construction = Construction(cup.get("construction", Construction.MUR.value))
            except ValueError as error:
                raise ValueError(
                    f"Ventouse {number}, construction doit valoir mur, tour ou porte."
                ) from error
            face = pymunk.Vec2d(1, 0).rotated(radians(angle % 360))
            ventouses.append(Ventouse(number, mount, (face.x, face.y), construction))

        planificateur = data.get("planificateur", "astar")
        if not isinstance(planificateur, str) or not planificateur:
            raise ValueError("planificateur doit être un identifiant non vide.")
        nom = data.get("nom", path.stem)
        if not isinstance(nom, str) or not nom:
            raise ValueError("nom doit être une chaîne non vide.")
        return RobotSpec(nom, contour, tuple(ventouses), _read_mouvement(data.get("mouvement")),
                         planificateur)
    except (OSError, ValueError, OverflowError) as error:
        raise ValueError(f"Configuration du robot {path.name} : {error}") from error


@cache
def load_robot(path: Path = ROBOT_FILE) -> RobotGeometry:
    return geometry_from_spec(load_robot_spec(path))
