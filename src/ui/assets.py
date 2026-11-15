"""Chargement des ressources d'affichage."""

from pathlib import Path

from ..infra.ressources import TAPIS


def charger_tapis(path: Path = TAPIS):
    from PIL import Image

    with Image.open(path) as image:
        return image.copy()
