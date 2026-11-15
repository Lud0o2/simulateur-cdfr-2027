"""Mutation des poses de dépôt."""

from copy import copy
from math import hypot, pi, remainder

import numpy as np

from ..strategies.poses import CENTERED_DROPS, DROP_ANGLES
from .mlp import DROP_OFFSET_LIMIT


def _mutate_drop_pose(network, rng):
    mutant = copy(network)
    wall = int(rng.integers(len(CENTERED_DROPS)))
    if rng.random() < 0.5:
        angles = list(network.drop_angles)
        choices = [angle for angle in DROP_ANGLES if remainder(angle - angles[wall], pi) != 0]
        angles[wall] = choices[int(rng.integers(len(choices)))]
        mutant.drop_angles = tuple(angles)
    else:
        offsets = list(network.drop_offsets)
        x, y = offsets[wall]
        choices = [(x + dx, y + dy) for step in (20, 40, 60, 100, 150, DROP_OFFSET_LIMIT)
                   for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step))
                   if hypot(x + dx, y + dy) <= DROP_OFFSET_LIMIT]
        if x or y:
            choices.append((0, 0))
        offsets[wall] = choices[int(rng.integers(len(choices)))]
        mutant.drop_offsets = tuple(offsets)
    return mutant


