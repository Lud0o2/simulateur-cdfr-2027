"""Entraîne et évalue le réseau de l'étape 5 sur des matchs simulés."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from copy import copy, deepcopy
import json
from math import cos, hypot, pi, remainder, sin
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from pymunk import Vec2d

from .match import (
    ALIGNED_ANGLES, CENTERED_DROPS, DROP_ANGLES, MATCH_SECONDS, REPLAY_DIR, Replay, _action_state, _clear, _drop_target,
    _pickup_approach, _shifted_drops, _trials, _wall_drop_zones, completion_beep, play_match, save_replay,
)
from .simulateur import Block, START_BLOCKS, TABLE_HEIGHT, TABLE_WIDTH, WALL_ZONES, Simulation, free_wall_slots, scoring_zones


LAYERS = (97, 16, 8, 1)
TABLE_DIAGONAL = hypot(TABLE_WIDTH, TABLE_HEIGHT)
ACTION_DISTANCE = 84
MODEL = REPLAY_DIR.parent / "reseau.npz"
PICKUP_LIMIT = 8
EXPLORATION_INTERVAL = 8
DROP_OFFSET_LIMIT = 200  # mm depuis le centre du rempart
CENTERED_OFFSETS = ((0, 0),) * len(CENTERED_DROPS)


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
            rows.append([*state, -1, *([0] * (LAYERS[0] - 80))])
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
            rotated = tuple(tuple(Vec2d(*corner).rotated(turn) for corner in shape) for shape in footprints)
            polygon = tuple(tuple(end + (Vec2d(*corner) - start).rotated(turn)) for corner in block.corners)
            # Évalue la pose finale. Le simulateur calcule le trajet seulement pour une pose essayée.
            valid = (_wall_drop_zones(Block(tuple(position), block.angle + turn, polygon))[1]
                     and _clear(end, end, rotated, obstacles))
            distance = start.get_distance(end) if valid else -TABLE_DIAGONAL
        mount, facing = simulation.geometry.mounts[cup - 1]
        rows.append([*state, pickup, *_pose(block.position, block.angle), distance / TABLE_DIAGONAL, *_pose(position, angle),
                     *(value / 380 for value in mount), *facing, duration / 8, gain / 8,
                     (position[0] - simulation.position[0]) / TABLE_WIDTH,
                     (position[1] - simulation.position[1]) / TABLE_HEIGHT])
    return np.asarray(rows, dtype=np.float32).reshape(-1, LAYERS[0])


def _candidate_count(choices, pickup):
    return min(PICKUP_LIMIT, len(choices)) if pickup else len(choices)


class Network:
    def __init__(self, rng):
        self.weights = [rng.normal(0, (2 / size) ** 0.5, (size, width)).astype(np.float32)
                        for size, width in zip(LAYERS, LAYERS[1:])]
        self.biases = [np.zeros(width, dtype=np.float32) for width in LAYERS[1:]]
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
        for layer in reversed(range(len(self.weights))):
            weights = self.weights[layer]
            previous = (gradient @ weights.T) * (values[layer] > 0) if layer else None
            weights -= learning_rate * (values[layer].T @ gradient)
            self.biases[layer] -= learning_rate * gradient.sum(axis=0)
            gradient = previous

    def train(self, inputs, targets, rng, *, learning_rate=0.01):
        for _ in range(64):
            batch = rng.integers(len(inputs), size=min(64, len(inputs)))
            values = self._forward(inputs[batch])
            gradient = 2 * (values[-1] - targets[batch, None]) / len(batch)
            self._learn(values, gradient, learning_rate)

    def imitate(self, examples, *, steps, learning_rate):
        inputs = np.concatenate([rows for rows, _ in examples])
        starts = np.cumsum([0, *(len(rows) for rows, _ in examples)])
        expected = starts[:-1] + [index for _, index in examples]
        for _ in range(steps):
            values = self._forward(inputs)
            logits = np.split(values[-1][:, 0], starts[1:-1])
            probabilities = [np.exp(group - group.max()) for group in logits]
            gradient = np.concatenate([group / group.sum() for group in probabilities])
            gradient[expected] -= 1
            self._learn(values, gradient[:, None] / len(examples), learning_rate)

    def _training_copy(self):
        candidate = copy(self)
        candidate.weights = [weights.copy() for weights in self.weights]
        candidate.biases = [biases.copy() for biases in self.biases]
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
        file = NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
        temporary = Path(file.name)
        try:
            with file:
                np.savez(file, *self.weights, *self.biases, drop_offsets=np.asarray(self.drop_offsets),
                         drop_angles=np.asarray(self.drop_angles),
                         experience=np.concatenate(experience) if experience else np.empty((0, LAYERS[0] + 5)),
                         training_matches=self.training_matches)
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
        return network


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


class _ReplayDemonstration:
    multiple_drops = False  # Les démonstrations annotées désignent une seule pose par rempart.

    def __init__(self, replay):
        pickups, drops = replay.get("prises"), replay.get("depots")
        if (not isinstance(pickups, list) or not isinstance(drops, list)
                or not pickups or len(pickups) != len(drops)):
            raise ValueError("Le replay doit décrire ses prises et ses dépôts.")
        batches = replay.get("tournees", [len(pickups)])
        if (not isinstance(batches, list) or not all(type(count) is int and count > 0 for count in batches)
                or sum(batches) != len(pickups)):
            raise ValueError("Tournées de démonstration invalides.")
        self.batches = iter(batches)
        self.remaining_pickups = next(self.batches)
        if not all(isinstance(pick, list) and len(pick) == 2 and all(type(n) is int and n > 0 for n in pick)
                   for pick in pickups):
            raise ValueError("Prises de démonstration invalides.")
        self.pickups = deque(map(tuple, pickups))
        self.drops = deque()
        self.drop_offsets = _read_drop_offsets(replay.get("decalages_depot_mm", CENTERED_OFFSETS))
        self.drop_angles = _read_drop_angles(replay.get("rotations_depot_rad", ALIGNED_ANGLES))
        drop_poses = _shifted_drops(self.drop_offsets, self.drop_angles)
        for drop in drops:
            if not (isinstance(drop, list) and len(drop) == 3 and all(type(n) is int for n in drop)):
                raise ValueError("Dépôt de démonstration invalide.")
            cup, wall, rotation = drop
            if cup < 1 or not 1 <= wall <= len(CENTERED_DROPS) or rotation not in (0, 180):
                raise ValueError("Dépôt de démonstration invalide.")
            site, axis = drop_poses[wall - 1]
            self.drops.append((cup, site, axis + (pi if rotation else 0)))
        self.examples = []
        self.observed = 0

    def order(self, simulation, held, choices, *, pickup, can_finish=False, pickup_plans=None):
        head = choices[:_candidate_count(choices, pickup)]
        if can_finish:
            head.append(None)
        if pickup:
            action = self.pickups.popleft() if self.remaining_pickups else None
            selected = next((choice for choice in head if choice is not None and
                             (choice[1] + 1, choice[2]) == action), None)
            if selected is None and action is not None:
                raise ValueError(f"Prise du replay inaccessible à {simulation.time:.2f} s.")
            if selected is not None:
                self.remaining_pickups -= 1
        else:
            if self.remaining_pickups or not self.drops:
                raise ValueError("Dépôt inattendu pendant la démonstration.")
            action = self.drops.popleft()
            selected = next((choice for choice in head if choice[2:] == action), None)
            if selected is None:
                raise ValueError(f"Dépôt du replay inaccessible à {simulation.time:.2f} s.")
            if len(held) == 1:
                self.remaining_pickups = next(self.batches, 0)
        if selected is None and not can_finish:
            raise ValueError("La démonstration ne peut pas terminer le chargement.")
        self.examples.append((features(simulation, held, head, pickup=pickup, pickup_plans=pickup_plans),
                              head.index(selected)))
        return [selected]

    def observe(self, _choice):
        self.observed += 1


def _replay_examples(replay):
    demonstration = Replay(replay)
    simulation = Simulation()
    policy = _ReplayDemonstration(replay)
    if len(demonstration.frames[0]["cups"]) != len(simulation.geometry.mounts):
        raise ValueError("Le replay utilise un autre nombre de ventouses. Il reste disponible en lecture.")
    result = play_match(simulation, policy=policy, record_frames=False)
    if (policy.pickups or policy.drops or policy.observed != len(policy.examples)
            or result["score"] != replay["score"] or result["score_retour"] != replay["score_retour"]):
        raise ValueError("Le simulateur ne reproduit pas la démonstration.")
    return policy.examples


def _learn_replay(network, replay):
    examples = _replay_examples(replay)
    best = network
    best_rank = _replay_rank(play_match(Simulation(), policy=MatchPolicy(best), record_frames=False))
    offsets = _read_drop_offsets(replay.get("decalages_depot_mm", CENTERED_OFFSETS))
    angles = _read_drop_angles(replay.get("rotations_depot_rad", ALIGNED_ANGLES))
    for rate in (0.025, 0.05, 0.075):
        candidate = deepcopy(network)
        candidate.drop_offsets = offsets
        candidate.drop_angles = angles
        for steps in (64, 64, 128, 256, 512):
            candidate.imitate(examples, steps=steps, learning_rate=rate)
            result = play_match(Simulation(), policy=MatchPolicy(candidate), record_frames=False)
            rank = _replay_rank(result)
            if rank > best_rank:
                best, best_rank = deepcopy(candidate), rank
    final = play_match(Simulation(), policy=MatchPolicy(best))
    if _replay_rank(final) != best_rank:
        raise ValueError("Le replay autonome ne reproduit pas la validation.")
    return best, final


def _replay_rank(replay):
    return replay["score"], -replay["duree"]


def _validated_replay(network, rank, seed):
    replay = play_match(Simulation(), policy=MatchPolicy(network))
    if _replay_rank(replay) != rank:
        raise ValueError("Le replay du réseau ne reproduit pas son résultat de validation.")
    replay.update(strategie="validation_reseau", graine=seed)
    return replay


def _simulate_match(job, *, record_frames=False):
    network, seed, warmup, resume, random_match = job
    rng = None if seed is None else np.random.default_rng(seed)
    policy = MatchPolicy(network, rng=rng, warmup=warmup, resume=resume, random_match=random_match)
    replay = play_match(Simulation(), policy=policy, record_frames=record_frames)
    if policy.opening and replay["score"] > replay["score_retour"]:
        replay["ouverture"] = tuple((policy.opening + [-1])[:2])
    if random_match:
        replay["exploration_aleatoire"] = True
    return replay, policy.history


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


def train_matches(count, seed, incumbent=None, *, resume=False, mutate_deposits=False):
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = min(count, available, 61 if os.name == "nt" else available)
    print(f"Simulation · {workers} processus pour {count} matchs d’entraînement", flush=True)
    with (ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool:
        return _train_matches(count, seed, incumbent, resume, workers, pool.map if pool else map, mutate_deposits)


def _train_matches(count, seed, incumbent, resume, workers, run_matches, mutate_deposits):
    rng = np.random.default_rng(seed)
    mutation_rng = np.random.default_rng(seed) if mutate_deposits else None
    network = incumbent if resume else Network(rng)
    network_rank = None
    experience = incumbent.experience.copy() if incumbent is not None else {}
    played = incumbent.training_matches if incumbent is not None and resume else 0
    learning_rate = 0.005 if resume else 0.01
    seeds = np.random.SeedSequence(seed, n_children_spawned=played).spawn(count)
    examples = deque(maxlen=512)
    exploration_rank = (-1, 0)
    exploration_job = None
    exploration_episode = None
    for start in range(0, count, workers):
        # La mémoire sert à l'apprentissage dans le processus principal.
        playing = network._for_simulation()
        jobs = [(playing, seeds[episode], not resume and episode < max(1, count // 3), resume,
                 (played + episode + 1) % EXPLORATION_INTERVAL == 0)
                for episode in range(start, min(start + workers, count))]
        # Tous les matchs du lot doivent partir des mêmes poids avant leur mise à jour.
        matches = list(run_matches(_simulate_match, jobs))
        for episode, (job, (replay, history)) in enumerate(zip(jobs, matches, strict=True), start + 1):
            rank = _replay_rank(replay)
            if rank > exploration_rank:
                exploration_rank = rank
                exploration_job = job
                exploration_episode = episode
            mode = "Exploration aléatoire" if replay.get("exploration_aleatoire") else "Exploration"
            print(f"{mode} {episode}/{count} · {rank[0]} points en {replay['duree']:.2f} s simulées · "
                  f"{len(history)} décisions", flush=True)

        candidates = []
        if mutation_rng is not None:
            candidates.extend(_mutate_drop_pose(network, mutation_rng) for _ in matches)
        remembered = tuple(experience.values())
        ranked = sorted((match for match in matches if match[1]), key=lambda match: _replay_rank(match[0]), reverse=True)
        for index, (replay, history) in enumerate(ranked):
            # Le temps départage les scores égaux avec un bonus de 0 à 0,5 point.
            time_bonus = 0.5 * (1 - replay["duree"] / MATCH_SECONDS)
            current = [(row, (replay["score"] - score + time_bonus) / 8) for row, score in history]
            examples.extend(current)
            # Le premier candidat apprend spécifiquement la meilleure trajectoire du lot.
            inputs, targets = zip(*(current if index == 0 else examples))
            inputs, targets = np.asarray(inputs), np.asarray(targets, dtype=np.float32)
            if remembered and (index > 0 or len(ranked) == 1):
                _, past = remembered[int(rng.integers(len(remembered)))]
                inputs = np.concatenate((inputs, past[:, :-1]))
                targets = np.concatenate((targets, past[:, -1]))
            candidate = network._training_copy()
            candidate.train(inputs, targets, rng, learning_rate=learning_rate)
            candidates.append(candidate)
            opening = replay.get("ouverture")
            if opening is not None:
                rank = _replay_rank(replay)
                if opening not in experience or rank > experience[opening][0]:
                    rows, rewards = zip(*current)
                    experience[opening] = rank, np.column_stack((np.asarray(rows), rewards)).astype(np.float32)

        if network_rank is None:
            candidates.insert(0, network)
        if not candidates:
            continue
        jobs = ((candidate._for_simulation(), None, False, False, False) for candidate in candidates)
        results = run_matches(_simulate_match, jobs)
        for candidate, (replay, _) in zip(candidates, results, strict=True):
            rank = _replay_rank(replay)
            if network_rank is None or rank > network_rank:
                network, network_rank = candidate, rank
        print(f"Validation · génération {start // workers + 1} · meilleur réseau "
              f"sans exploration {network_rank[0]} points en {-network_rank[1]:.2f} s · "
              f"record avec exploration {exploration_rank[0]} points en {-exploration_rank[1]:.2f} s", flush=True)

    trained_replay = _validated_replay(network, network_rank, seed)
    winner_replay = trained_replay
    if incumbent is not None and not resume:
        replay, _ = _simulate_match((incumbent, None, False, False, False))
        rank = _replay_rank(replay)
        if rank >= network_rank:
            network = incumbent
            winner_replay = _validated_replay(incumbent, rank, seed)
    network.experience = experience
    network.training_matches = (incumbent.training_matches if incumbent is not None else 0) + count
    # Les poids du match et sa graine suffisent à refaire les images de l'essai retenu.
    exploration_replay, _ = _simulate_match(exploration_job, record_frames=True)
    if _replay_rank(exploration_replay) != exploration_rank:
        raise ValueError("Le replay avec exploration ne reproduit pas son résultat d’entraînement.")
    exploration_replay.update(strategie="entrainement_reseau", graine=seed, match=exploration_episode)
    return network, winner_replay, trained_replay, exploration_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Entraîne le réseau sur la table officielle.")
    train.add_argument("--matches", type=int, default=12)
    train.add_argument("--seed", type=int, default=7, help="Graine des tirages (par défaut : %(default)s).")
    train.add_argument("--resume", action="store_true", help="Repart des poids du fichier choisi.")
    train.add_argument("--mutate-deposits", action="store_true", help="Teste des positions et des angles de dépôt.")
    imitate = commands.add_parser("imitate", help="Apprend d’un replay annoté et valide le réseau en match autonome.")
    imitate.add_argument("number", type=int, help="Numéro du replay de démonstration.")
    commands.add_parser("compare", help="Compare le réseau sauvegardé à l’étape 4 sur la table officielle.")
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.matches < 1:
                parser.error("--matches doit être strictement positif.")
            incumbent = Network.load(args.model) if args.resume or args.model.exists() else None
            if args.resume:
                print(f"Reprise des poids de {args.model.resolve()}", flush=True)
            network, winner, trained, explored = train_matches(args.matches, args.seed, incumbent, resume=args.resume,
                                                                mutate_deposits=args.mutate_deposits)
            network.save(args.model)
            print("Poids conservés" if network is incumbent else "Poids sauvegardés")
            print(f"Mémoire · {len(network.experience)} ouvertures")
            replays = [(f"Exploration {explored['match']}/{args.matches}", explored)]
            if trained is not winner:
                replays.append(("Réseau de la session", trained))
            replays.append(("Réseau sauvegardé", winner))
            for label, replay in replays:
                number, _ = save_replay(replay)
                print(f"{label} · {replay['score']} points en {replay['duree']:.2f} s · replay {number}")
            completion_beep()
            return

        if args.command == "imitate":
            path = _trials(REPLAY_DIR).get(args.number)
            if path is None:
                parser.error(f"Essai {args.number} introuvable.")
            replay = json.loads(path.read_text(encoding="utf-8"))
            incumbent = Network.load(args.model)
            network, validation = _learn_replay(incumbent, replay)
            validation.update(strategie="imitation_reseau", demonstration=args.number,
                              score_reference=replay["score"], duree_reference=replay["duree"])
            number, _ = save_replay(validation)
            if network is not incumbent:
                network.save(args.model)
            else:
                print(f"Poids existants conservés dans {args.model}")
            print(f"Replay {args.number} · {replay['score']} points en {replay['duree']:.2f} s")
            print(f"Réseau autonome · {validation['score']} points en {validation['duree']:.2f} s "
                  f"(+{validation['score_retour']} au retour)")
            print(f"python -B -m src.match replay {number}", flush=True)
            completion_beep()
            return

        network = Network.load(args.model)
        reference = play_match(Simulation(), record_frames=False)
        replay = play_match(Simulation(), policy=MatchPolicy(network))
        score = replay["score"]
        replay.update(strategie="reseau", score_reference=reference["score"])
        number, path = save_replay(replay)
        print(f"Table officielle · étape 4 {reference['score']} points · réseau {score} points "
              f"(+{replay['score_retour']} au retour)")
        print(f"Durées simulées · étape 4 {reference['duree']:.2f} s · réseau {replay['duree']:.2f} s")
        print(f"Replay de l’essai {number} · {path}")
        print(f"python -B -m src.match replay {number}", flush=True)
        completion_beep()
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
