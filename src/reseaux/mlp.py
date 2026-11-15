"""Réseau MLP NumPy avec Adam et persistance .npz."""

from copy import copy
from math import hypot, pi
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ..strategies.poses import ALIGNED_ANGLES, CENTERED_DROPS
from ..terrains.officiel_2027 import START_BLOCKS
from .base import ACTION_DISTANCE, LAYERS


BATCH_SIZE = 64
EPOCHS = 8
MAX_STEPS = 64
LEARNING_RATE = 0.002
LEARNING_RATE_FLOOR = 0.0002
LEARNING_RATE_HALVING = 200
DROP_OFFSET_LIMIT = 200
CENTERED_OFFSETS = ((0, 0),) * len(CENTERED_DROPS)
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
MOMENT_NAMES = ("moment_w", "variance_w", "moment_b", "variance_b")


def _learning_rate(training_matches):
    # Un réseau déjà éprouvé bouge moins qu'une initialisation aléatoire.
    return max(LEARNING_RATE_FLOOR, LEARNING_RATE * 0.5 ** (max(0, training_matches) // LEARNING_RATE_HALVING))


def _read_drop_offsets(values):
    if (not isinstance(values, (list, tuple)) or len(values) != len(CENTERED_DROPS)
            or any(not isinstance(offset, (list, tuple)) or len(offset) != 2
                   or not all(type(value) in (int, float) and abs(value) <= DROP_OFFSET_LIMIT for value in offset)
                   or hypot(*offset) > DROP_OFFSET_LIMIT for offset in values)):
        raise ValueError("Décalages de dépôt invalides.")
    return tuple(map(tuple, values))


def _read_drop_angles(values):
    if (not isinstance(values, (list, tuple)) or len(values) != len(ALIGNED_ANGLES)
            or not all(type(angle) in (int, float) and abs(angle) <= pi / 2 for angle in values)):
        raise ValueError("Angles de dépôt invalides.")
    return tuple(values)


class Network:
    def __init__(self, rng):
        self.weights = [rng.normal(0, (2 / size) ** 0.5, (size, width)).astype(np.float32)
                        for size, width in zip(LAYERS, LAYERS[1:])]
        self.biases = [np.zeros(width, dtype=np.float32) for width in LAYERS[1:]]
        self.moments = [[np.zeros_like(weights), np.zeros_like(weights),
                         np.zeros_like(biases), np.zeros_like(biases)]
                        for weights, biases in zip(self.weights, self.biases)]
        self.updates = 0
        self.drop_offsets = CENTERED_OFFSETS
        self.drop_angles = ALIGNED_ANGLES
        self.experience = {}
        self.training_matches = 0

    def _forward(self, inputs):
        values = [inputs]
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            layer = values[-1] @ weights + biases
            values.append(np.maximum(layer, 0) if index < len(self.weights) - 1 else layer)
        return values

    def predict(self, inputs):
        return self._forward(inputs)[-1][:, 0]

    def _learn(self, values, gradient, learning_rate):
        self.updates += 1
        first = 1 - ADAM_BETAS[0] ** self.updates
        second = 1 - ADAM_BETAS[1] ** self.updates
        for layer in reversed(range(len(self.weights))):
            weights = self.weights[layer]
            previous = (gradient @ weights.T) * (values[layer] > 0) if layer else None
            moment, variance, bias_moment, bias_variance = self.moments[layer]
            self._apply(weights, moment, variance, values[layer].T @ gradient, learning_rate, first, second)
            self._apply(self.biases[layer], bias_moment, bias_variance, gradient.sum(axis=0),
                        learning_rate, first, second)
            gradient = previous

    @staticmethod
    def _apply(parameter, moment, variance, gradient, learning_rate, first, second):
        # Adam normalise chaque paramètre par la moyenne et la variance de son gradient.
        moment *= ADAM_BETAS[0]
        moment += (1 - ADAM_BETAS[0]) * gradient
        variance *= ADAM_BETAS[1]
        variance += (1 - ADAM_BETAS[1]) * gradient * gradient
        parameter -= learning_rate * (moment / first) / (np.sqrt(variance / second) + ADAM_EPSILON)

    def train(self, inputs, targets, rng, *, learning_rate=LEARNING_RATE):
        batch = min(BATCH_SIZE, len(inputs))
        # Un budget en époques évite de mémoriser un lot de quelques lignes.
        steps = max(1, min(MAX_STEPS, round(EPOCHS * len(inputs) / batch)))
        for _ in range(steps):
            sample = rng.integers(len(inputs), size=batch)
            values = self._forward(inputs[sample])
            gradient = 2 * (values[-1] - targets[sample, None]) / len(sample)
            self._learn(values, gradient, learning_rate)

    def _training_copy(self):
        candidate = copy(self)
        candidate.weights = [weights.copy() for weights in self.weights]
        candidate.biases = [biases.copy() for biases in self.biases]
        candidate.moments = [[values.copy() for values in layer] for layer in self.moments]
        return candidate

    def _for_simulation(self):
        if not self.experience:
            return self
        network = copy(self)
        network.experience = {}
        return network

    def save(self, path):
        # Deux indices d'ouverture, le score, le temps négatif, puis les entrées et la cible.
        experience = [np.column_stack((np.broadcast_to((*opening, *rank), (len(rows), 4)), rows))
                      for opening, (rank, rows) in self.experience.items()]
        moments = {f"adam_{name}_{index}": value
                   for index, layer in enumerate(self.moments)
                   for name, value in zip(MOMENT_NAMES, layer)}
        file = NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
        temporary = Path(file.name)
        try:
            with file:
                np.savez(file, *self.weights, *self.biases, drop_offsets=np.asarray(self.drop_offsets),
                         drop_angles=np.asarray(self.drop_angles),
                         experience=np.concatenate(experience) if experience else np.empty((0, LAYERS[0] + 5)),
                         training_matches=self.training_matches, adam_updates=self.updates, **moments)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        network = cls(np.random.default_rng(0))
        with np.load(path, allow_pickle=False) as data:
            presence_bias = 0
            for index, values in enumerate((*network.weights, *network.biases)):
                loaded = data[f"arr_{index}"]
                if index == 0 and loaded.shape == (LAYERS[0] - 1, LAYERS[1]):
                    if loaded.dtype != values.dtype or not np.all(np.isfinite(loaded)):
                        raise ValueError("Paramètres du réseau invalides.")
                    # Les anciennes lignes de présence sont reprises par le biais.
                    presence_bias = loaded[11:77:7].sum(axis=0)
                    loaded = np.insert(loaded, ACTION_DISTANCE, 0, axis=0)
                    loaded[11:77:7] = 0
                    loaded[78] /= 2
                if loaded.shape != values.shape or loaded.dtype != values.dtype or not np.all(np.isfinite(loaded)):
                    raise ValueError("Paramètres du réseau invalides.")
                values[:] = loaded
            network.biases[0] += presence_bias
            if not np.all(np.isfinite(network.biases[0])):
                raise ValueError("Paramètres du réseau invalides.")
            if "drop_offsets" in data:
                network.drop_offsets = _read_drop_offsets(data["drop_offsets"].tolist())
            if "drop_angles" in data:
                network.drop_angles = _read_drop_angles(data["drop_angles"].tolist())
            if "experience" in data:
                rows = data["experience"]
                if (rows.ndim != 2 or rows.shape[1] != LAYERS[0] + 5 or rows.dtype != np.float64
                        or not np.all(np.abs(rows) <= np.finfo(np.float32).max)
                        or not np.all((rows[:, :2] == np.trunc(rows[:, :2]))
                                      & (rows[:, :2] >= -1) & (rows[:, :2] < len(START_BLOCKS)))):
                    raise ValueError("Mémoire d’entraînement invalide.")
                for opening in np.unique(rows[:, :2], axis=0):
                    group = rows[np.all(rows[:, :2] == opening, axis=1)]
                    network.experience[tuple(map(int, opening))] = (tuple(group[0, 2:4]), group[:, 4:].astype(np.float32))
            if "training_matches" in data:
                count = data["training_matches"].tolist()
                if type(count) is not int or count < 0:
                    raise ValueError("Compteur d’entraînement invalide.")
                network.training_matches = count
            for index, layer in enumerate(network.moments):
                for name, target in zip(MOMENT_NAMES, layer):
                    key = f"adam_{name}_{index}"
                    if key not in data:
                        continue
                    loaded = data[key]
                    if loaded.shape != target.shape or loaded.dtype != target.dtype or not np.all(np.isfinite(loaded)):
                        raise ValueError("Paramètres du réseau invalides.")
                    target[:] = loaded
            if "adam_updates" in data:
                count = data["adam_updates"].tolist()
                if type(count) is not int or count < 0:
                    raise ValueError("Compteur d’entraînement invalide.")
                network.updates = count
        return network


