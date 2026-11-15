"""Valeurs de score du règlement Eurobot 2027, § E.1."""

from dataclasses import dataclass

from ..domaine.robot import Construction


@dataclass(frozen=True)
class Bareme2027:
    points_pierre: int = 1
    points_pile: int = 3
    points_mur: int = 5
    points_tour: int = 5
    points_porte: int = 20
    points_retour_complet: int = 10
    points_retour_partiel: int = 5

    def points_construction(self, construction: Construction) -> int:
        return {
            Construction.MUR: self.points_mur,
            Construction.TOUR: self.points_tour,
            Construction.PORTE: self.points_porte,
        }[construction]
