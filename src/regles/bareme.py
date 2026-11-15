"""Valeurs de score du règlement Eurobot 2027."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Bareme2027:
    points_pile: int = 3
    points_mur: int = 5
    points_retour_complet: int = 10
    points_retour_partiel: int = 5
    # TODO : ajouter les valeurs des tours et des portes avec la section du règlement.
