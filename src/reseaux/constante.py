"""Réseau constant : baseline et test de contrat."""

from pathlib import Path

import numpy as np

from ..strategies.poses import ALIGNED_ANGLES, CENTERED_DROPS


class Constante:
    id = "constante"

    def __init__(self, valeur: float = 0.0, input_size: int = 97):
        self.valeur = float(valeur)
        self.input_size = input_size
        self.drop_offsets = CENTERED_DROPS
        self.drop_angles = ALIGNED_ANGLES

    def predict(self, inputs):
        return np.full(len(inputs), self.valeur, dtype=np.float32)

    def train(self, inputs, targets, rng, *, learning_rate=0.002):
        return None

    def save(self, path: Path) -> None:
        np.savez(path, valeur=np.asarray(self.valeur),
                 drop_offsets=np.asarray(self.drop_offsets), drop_angles=np.asarray(self.drop_angles))

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            return cls(float(data["valeur"]))
