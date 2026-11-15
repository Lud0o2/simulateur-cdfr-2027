"""Validation de valeurs numériques venues de fichiers ou de l'interface."""

from math import isfinite

from .modele import Point


def _finite_number(value):
    return type(value) in (int, float) and isfinite(value)


def _read_point(value, field: str) -> Point:
    if not isinstance(value, list) or len(value) != 2 or not all(map(_finite_number, value)):
        raise ValueError(f"{field} doit contenir deux coordonnées finies en mm.")
    return value[0], value[1]
