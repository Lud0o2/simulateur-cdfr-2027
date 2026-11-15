"""Persistance des modèles."""

from pathlib import Path


def sauver_modele(network, path: Path) -> None:
    network.save(path)


def charger_modele(path: Path):
    from .mlp import Network

    return Network.load(path)
