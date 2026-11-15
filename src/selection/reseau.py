"""Politique de sélection guidée par un réseau."""

import numpy as np

from ..reseaux.caracteristiques import _candidate_count, features
from ..reseaux.base import PICKUP_LIMIT
from ..reseaux.mlp import CENTERED_OFFSETS
from ..strategies.poses import ALIGNED_ANGLES, CENTERED_DROPS


class MatchPolicy:
    def __init__(self, network, *, rng=None, warmup=False, resume=False, random_match=False):
        self.network = network
        self.rng = rng
        self.warmup = warmup
        self.resume = resume
        self.history = []
        self.opening = []
        self.random_match = random_match
        self.drop_offsets = getattr(network, "drop_offsets", CENTERED_OFFSETS)
        self.drop_angles = getattr(network, "drop_angles", ALIGNED_ANGLES)

    def order(self, simulation, held, choices, *, pickup, can_finish=False, pickup_plans=None):
        # shortcut: le réseau classe les huit prises les plus rapides, à élargir si elles bloquent l'exploration.
        limit = len(choices) if self.random_match else _candidate_count(choices, pickup)
        head = choices[:limit]
        if can_finish and (held or not self.random_match):
            head.append(None)
        inputs = features(simulation, held, head, pickup=pickup, pickup_plans=pickup_plans)
        if self.rng is not None:
            self.rows = dict(zip(head, inputs))
            self.score_before = simulation.score
            self.pickup = pickup
        if self.random_match:
            return [head[index] for index in self.rng.permutation(len(head))]
        order = np.argsort(-self.network.predict(inputs), kind="stable")
        if self.rng is not None:
            if self.resume:
                if len(order) > 1 and self.rng.random() < 0.30:
                    alternatives = order[1:5].tolist()
                    if pickup and held:
                        preferred = head[order[0]]
                        fastest = next((index for index, choice in enumerate(head)
                                        if choice is not None and (preferred is None or choice[1] != preferred[1])),
                                       None)
                        if fastest is not None and fastest not in alternatives:
                            alternatives[-1] = fastest
                    alternate = self.rng.choice(alternatives)
                    rank = int(np.flatnonzero(order == alternate)[0])
                    order[0], order[rank] = alternate, order[0]
            elif self.rng.random() < 0.25:
                order = self.rng.permutation(len(head))
            elif self.warmup or self.rng.random() < 0.2:
                return choices
        return [head[index] for index in order] + choices[limit:]

    def observe(self, choice):
        if self.rng is not None and choice in self.rows:
            self.history.append((self.rows[choice], self.score_before))
            if self.pickup and choice is not None and len(self.opening) < 2:
                self.opening.append(choice[1])


