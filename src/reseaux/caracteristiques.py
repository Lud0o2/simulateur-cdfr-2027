"""Encodage des observations et des candidats pour un réseau."""

from math import cos, sin

import numpy as np
from pymunk import Vec2d

from ..actions.approche import _drop_target, _pickup_approach, _wall_drop_zones
from ..actions.execution import _action_state
from ..domaine.modele import Block
from ..planification.astar import _rotated_footprint, _visibility
from ..regles.phases import MATCH_SECONDS
from ..regles.score_2027 import free_wall_slots, scoring_zones
from ..terrains.officiel_2027 import START_BLOCKS, TABLE_HEIGHT, TABLE_WIDTH, WALL_ZONES
from .base import ACTION_DISTANCE, LAYERS, PICKUP_LIMIT, TABLE_DIAGONAL


def _pose(position, angle):
    x, y = position
    return x / TABLE_WIDTH, y / TABLE_HEIGHT, cos(angle), sin(angle)


def features(simulation, held, choices, *, pickup, pickup_plans=None):
    blocks = simulation.blocks
    carried = set(held.values())
    state = [*_pose(simulation.position, simulation.heading),
             (MATCH_SECONDS - simulation.time) / MATCH_SECONDS, simulation.score / 80,
             len(held) / len(simulation.geometry.mounts)]
    built = 0
    for index, block in enumerate(blocks[:10]):
        is_held = index in carried
        in_zone = not is_held and scoring_zones(block)[1]
        x, y = block.position
        in_stock = not is_held and any(left <= x <= right and top <= y <= bottom
                                       for left, top, right, bottom in START_BLOCKS)
        built += in_zone
        state.extend((*_pose(block.position, block.angle), in_stock, is_held, in_zone))
    state.extend([0] * (7 * max(0, 10 - len(blocks))))
    slots = free_wall_slots(block for index, block in enumerate(blocks) if index not in carried)
    state.extend(((len(blocks) - len(held) - built) / 10, (len(WALL_ZONES) - slots) / len(WALL_ZONES)))

    rows = []
    action_state = None
    for choice in choices:
        if choice is None:
            # Termine le chargement si une ventouse est occupée, ou demande le retour.
            rows.append([-1, *([0] * (LAYERS[0] - len(state) - 1))])
            continue
        if action_state is None:
            action_state = _action_state(simulation)
        start, footprints, obstacles = action_state
        if pickup:
            duration, index, cup = choice
            block = blocks[index]
            position, angle = block.position, block.angle
            gain = 0
            plan = (pickup_plans[index, cup] if pickup_plans is not None
                    else _pickup_approach(simulation, block, cup, action_state))
            distance = plan.distance_from(start) if plan is not None else -TABLE_DIAGONAL
        else:
            gain, duration, cup, position, angle = choice
            block = blocks[held[cup]]
            end, turn = _drop_target(block, Vec2d(*position), angle, start)
            polygon = tuple(tuple(end + (Vec2d(*corner) - start).rotated(turn)) for corner in block.corners)
            # Évalue la pose finale. Le simulateur calcule le trajet seulement pour une pose essayée.
            distance = -TABLE_DIAGONAL
            if _wall_drop_zones(Block(tuple(position), block.angle + turn, polygon))[1]:
                rotated = tuple(_rotated_footprint(shape, turn) for shape in footprints)
                if _visibility(rotated, obstacles, 0)(end, end):
                    distance = start.get_distance(end)
        mount, facing = simulation.geometry.mounts[cup - 1]
        rows.append([pickup, *_pose(block.position, block.angle), distance / TABLE_DIAGONAL, *_pose(position, angle),
                     *(value / 380 for value in mount), *facing, duration / 8, gain / 8,
                     (position[0] - simulation.position[0]) / TABLE_WIDTH,
                     (position[1] - simulation.position[1]) / TABLE_HEIGHT])
    inputs = np.empty((len(rows), LAYERS[0]), dtype=np.float32)
    inputs[:, :len(state)] = state
    inputs[:, len(state):] = np.asarray(rows, dtype=np.float32).reshape(len(rows), LAYERS[0] - len(state))
    return inputs


def _candidate_count(choices, pickup):
    return min(PICKUP_LIMIT, len(choices)) if pickup else len(choices)




class EncodeurCaracteristiques:
    schema = "match-2027-v1"

    def __init__(self, fonction=features):
        self.fonction = fonction

    def encoder(self, *args, **kwargs):
        return self.fonction(*args, **kwargs)
