"""Réseau linéaire minimal, interchangeable avec le MLP."""

from pathlib import Path

import numpy as np

from ..strategies.poses import ALIGNED_ANGLES, CENTERED_DROPS


class Lineaire:
    id = "lineaire"

    def __init__(self, rng, input_size: int = 97):
        self.weights = rng.normal(0, 0.01, (input_size, 1)).astype(np.float32)
        self.bias = np.zeros(1, dtype=np.float32)
        self.drop_offsets = CENTERED_DROPS
        self.drop_angles = ALIGNED_ANGLES

    def predict(self, inputs):
        return (inputs @ self.weights + self.bias)[:, 0]

    def train(self, inputs, targets, rng, *, learning_rate=0.002):
        gradient = (inputs.T @ (self.predict(inputs) - targets))[:, None] / max(len(inputs), 1)
        self.weights -= learning_rate * gradient.astype(np.float32)
        self.bias -= learning_rate * float(np.mean(self.predict(inputs) - targets))

    def save(self, path: Path) -> None:
        np.savez(path, weights=self.weights, bias=self.bias,
                 drop_offsets=np.asarray(self.drop_offsets), drop_angles=np.asarray(self.drop_angles))

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            network = cls(np.random.default_rng(0), data["weights"].shape[0])
            network.weights = data["weights"].astype(np.float32)
            network.bias = data["bias"].astype(np.float32)
        return network
