"""Sélection aléatoire reproductible."""


class Aleatoire:
    id = "aleatoire"

    def __init__(self, rng):
        self.rng = rng

    def classer(self, observation, candidats):
        ordonnes = list(candidats)
        self.rng.shuffle(ordonnes)
        return ordonnes

    def observer(self, candidat, valeur: float) -> None:
        return None
