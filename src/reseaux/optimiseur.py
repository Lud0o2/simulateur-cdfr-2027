"""Optimiseur Adam indépendant du modèle."""

import numpy as np

from .mlp import ADAM_BETAS, ADAM_EPSILON


class OptimiseurAdam:
    def __init__(self, parameters, *, betas=ADAM_BETAS, epsilon=ADAM_EPSILON):
        self.betas = betas
        self.epsilon = epsilon
        self.updates = 0
        self.moments = [[np.zeros_like(parameter), np.zeros_like(parameter)] for parameter in parameters]

    def pas(self, index: int, parameter, gradient, learning_rate: float) -> None:
        self.updates += 1
        first = 1 - self.betas[0] ** self.updates
        second = 1 - self.betas[1] ** self.updates
        moment, variance = self.moments[index]
        moment *= self.betas[0]
        moment += (1 - self.betas[0]) * gradient
        variance *= self.betas[1]
        variance += (1 - self.betas[1]) * gradient * gradient
        parameter -= learning_rate * (moment / first) / (np.sqrt(variance / second) + self.epsilon)
