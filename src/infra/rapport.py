"""Reporting d'entraînement interchangeable."""

from typing import Protocol


class Rapporteur(Protocol):
    def ligne(self, message: str, **champs) -> None: ...


class RapporteurConsole:
    def ligne(self, message: str, **champs) -> None:
        print(message, flush=True)


class RapporteurSilencieux:
    def ligne(self, message: str, **champs) -> None:
        return None


class RapporteurListe:
    def __init__(self):
        self.lignes = []

    def ligne(self, message: str, **champs) -> None:
        self.lignes.append((message, champs))
