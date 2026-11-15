"""Mécanique de score, indépendante des valeurs d'un barème précis."""

from collections.abc import Iterable
from typing import Protocol


class Bareme(Protocol):
    points_pile: int
    points_mur: int


def construction_score(sites: Iterable[tuple[tuple[int, ...], bool]], bareme: Bareme) -> int:
    stones = 0
    walls: dict[int, tuple[int, ...]] = {}

    def assign_wall(zones: tuple[int, ...], visited: set[int]) -> bool:
        # Réattribue un mur déjà compté si deux piles se disputent la même zone.
        for zone in zones:
            if zone in visited:
                continue
            visited.add(zone)
            if zone not in walls or assign_wall(walls[zone], visited):
                walls[zone] = zones
                return True
        return False

    for zones, scoring in sites:
        stones += bareme.points_pile * scoring
        assign_wall(zones, set())
    return stones + bareme.points_mur * len(walls)
