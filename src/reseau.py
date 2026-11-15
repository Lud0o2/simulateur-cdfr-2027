"""Entraîne et évalue le réseau de l'étape 5 sur des matchs simulés."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from copy import copy
from math import cos, hypot, pi, remainder, sin
import os
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile
from time import monotonic

import numpy as np
from pymunk import Vec2d

from .match import (
    ALIGNED_ANGLES, CENTERED_DROPS, DROP_ANGLES, MATCH_SECONDS, REPLAY_DIR, _action_state, _drop_target,
    _pickup_approach, _rotated_footprint, _visibility, _wall_drop_zones,
    completion_beep, play_match, save_replay,
)
from .simulateur import Block, START_BLOCKS, TABLE_HEIGHT, TABLE_WIDTH, WALL_ZONES, Simulation, free_wall_slots, scoring_zones


LAYERS = (97, 16, 8, 1)
TABLE_DIAGONAL = hypot(TABLE_WIDTH, TABLE_HEIGHT)
ACTION_DISTANCE = 84
MODEL = REPLAY_DIR.parent / "reseau.npz"
PICKUP_LIMIT = 8
EXPLORATION_INTERVAL = 8
PROGRESS_WINDOW = 50  # matchs d'entraînement sans nouveau record autonome
DROP_OFFSET_LIMIT = 200  # mm depuis le centre du rempart
CENTERED_OFFSETS = ((0, 0),) * len(CENTERED_DROPS)
BATCH_SIZE = 64
EPOCHS = 8  # passages sur les exemples mémorisés par mise à jour
MAX_STEPS = 64
LEARNING_RATE = 0.002
LEARNING_RATE_FLOOR = 0.0002
LEARNING_RATE_HALVING = 200  # matchs d'entraînement entre deux divisions du pas
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
        # Deux indices d'ouverture, les points par seconde, le score, puis les entrées et la cible.
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
                         rank_version=np.int64(2),
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
                    rank = tuple(group[0, 2:4])
                    if "rank_version" not in data:
                        # Ancien classement : score puis temps négatif.
                        rank = _match_rank(rank[0], -rank[1])
                    network.experience[tuple(map(int, opening))] = (rank, group[:, 4:].astype(np.float32))
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


def _match_rank(score, duration):
    """Classe un match par points par seconde, puis par total de points."""
    # Une durée nulle donnerait une cadence infinie : le plancher la garde en tête.
    return score / max(duration, 1e-6), score


def _replay_rank(replay):
    return _match_rank(replay["score"], replay["duree"])


def _format_rank(rank):
    return f"{rank[0]:.2f} pts/s · {rank[1]} points"


def _format_elapsed(seconds):
    hours, rest = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {seconds:02d} s"
    return f"{seconds} s"


def _validated_replay(network, rank, seed):
    replay = play_match(Simulation(), policy=MatchPolicy(network))
    if _replay_rank(replay) != rank:
        raise ValueError("Le replay du réseau ne reproduit pas son résultat de validation.")
    replay.update(strategie="validation_reseau", graine=seed)
    return replay


def _simulate_match(job):
    network, seed, warmup, resume, random_match = job
    rng = None if seed is None else np.random.default_rng(seed)
    policy = MatchPolicy(network, rng=rng, warmup=warmup, resume=resume, random_match=random_match)
    replay = play_match(Simulation(), policy=policy, record_frames=False)
    if policy.opening and replay["score"] > replay["score_retour"]:
        replay["ouverture"] = tuple((policy.opening + [-1])[:2])
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


def _training_workers(count):
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return min(count, available, 61 if os.name == "nt" else available)


def train_matches(count, seed, incumbent=None, *, resume=False, learning_rate=None):
    workers = _training_workers(count)
    print(f"Simulation · {workers} processus pour {count} matchs d’entraînement", flush=True)
    with (ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool:
        network, rank = _train_matches(count, seed, incumbent, resume, workers, pool.map if pool else map,
                                       learning_rate=learning_rate)
    replay = _validated_replay(network, rank, seed)
    if incumbent is not None and not resume:
        previous, _ = _simulate_match((incumbent, None, False, False, False))
        if _replay_rank(previous) >= rank:
            incumbent.experience = network.experience
            incumbent.training_matches = network.training_matches
            network = incumbent
    return network, replay


def _train_endless(seed, model, incumbent, *, resume, learning_rate=None):
    started = monotonic()
    saved_rank = None
    saved_duration = None
    improved_at = None
    session_best = None
    series_best = None
    series_replays = {}
    series = 1
    seeds = np.random.SeedSequence(seed)
    workers = _training_workers(PROGRESS_WINDOW)

    def record(network, rank, duration):
        nonlocal saved_rank, saved_duration, improved_at, session_best, series_best
        if series_best is None or rank > series_best[1]:
            series_best = (network, rank, duration)
        if session_best is None or rank > session_best[0]:
            session_best = rank, series, duration
        if saved_rank is None or rank > saved_rank:
            network.save(model)
            saved_rank, saved_duration = rank, duration
            improved_at = monotonic()

    def save_series():
        nonlocal series_best
        if series_best is None:
            return
        network, rank, duration = series_best
        replay = _validated_replay(network, rank, seed)
        replay.update(strategie="entrainement_endless", serie=series)
        number, _ = save_replay(replay, descriptive=True)
        series_replays[series] = number
        series_best = None
        print(f"Série {series} · {_format_rank(rank)} en {duration:.2f} s · replay {number}", flush=True)

    def status(*, generation, played, batch, batch_duration, stalled):
        rank = saved_rank if saved_rank is not None else batch
        duration = saved_duration if saved_rank is not None else batch_duration
        since = (f"aucune amélioration depuis {_format_elapsed(monotonic() - started)}"
                 if improved_at is None else f"amélioration il y a {_format_elapsed(monotonic() - improved_at)}")
        print(f"Série {series} · génération {generation} · {played} matchs"
              f" · record {_format_rank(rank)} en {duration:.2f} s"
              f" · meilleur du lot {_format_rank(batch)} en {batch_duration:.2f} s"
              f" · {since} · stagnation {stalled}/{PROGRESS_WINDOW}", flush=True)

    try:
        if incumbent is not None:
            replay, _ = _simulate_match((incumbent, None, False, False, False))
            saved_rank = _replay_rank(replay)
            saved_duration = replay["duree"]
        with (ProcessPoolExecutor(max_workers=workers, initializer=signal.signal,
                                  initargs=(signal.SIGINT, signal.SIG_IGN)) if workers > 1 else nullcontext()) as pool:
            run_matches = pool.map if pool else map
            starting_network = incumbent if resume else None
            while True:
                print(f"Série {series} · graine {seed} · "
                      f"{'reprise du modèle' if starting_network is not None else 'nouveau réseau aléatoire'}",
                      flush=True)
                _train_matches(None, seed, starting_network, starting_network is not None,
                               workers, run_matches, on_record=record, on_generation=status,
                               learning_rate=learning_rate)
                save_series()
                series += 1
                seed = int(seeds.spawn(1)[0].generate_state(1)[0])
                starting_network = None
    except KeyboardInterrupt:
        # La série interrompue garde aussi son meilleur résultat.
        save_series()
        if session_best is None:
            print("Entraînement arrêté.", flush=True)
            return
        rank, best_series, duration = session_best
        print(f"Session · {_format_rank(rank)} en {duration:.2f} s"
              f" · série {best_series} · replay {series_replays[best_series]}", flush=True)


def _train_matches(count, seed, incumbent, resume, workers, run_matches, *, on_record=None,
                   on_generation=None, learning_rate=None):
    endless = count is None
    rng = np.random.default_rng(seed)
    mutation_rng = np.random.default_rng(seed)
    network = incumbent if resume else Network(rng)
    network_rank = None
    network_duration = None
    stalled = 0
    generation = 0
    experience = incumbent.experience.copy() if incumbent is not None else {}
    previous_matches = incumbent.training_matches if incumbent is not None else 0
    played = previous_matches if resume else 0
    learning_rate = _learning_rate(played) if learning_rate is None else learning_rate
    warmup_matches = max(1, (PROGRESS_WINDOW if endless else count) // 3)
    seeds = np.random.SeedSequence(seed, n_children_spawned=played)
    examples = deque(maxlen=512)
    start = 0
    while endless or start < count:
        generation += 1
        remaining = PROGRESS_WINDOW - stalled if endless else count - start
        size = min(workers, remaining)
        stop = start + size
        # La mémoire sert à l'apprentissage dans le processus principal.
        playing = network._for_simulation()
        jobs = [(playing, match_seed, not resume and episode < warmup_matches, resume,
                 (played + episode + 1) % EXPLORATION_INTERVAL == 0)
                for episode, match_seed in enumerate(seeds.spawn(size), start)]
        # Tous les matchs du lot doivent partir des mêmes poids avant leur mise à jour.
        matches = list(run_matches(_simulate_match, jobs))

        candidates = [_mutate_drop_pose(network, mutation_rng) for _ in matches]
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
        previous_rank = network_rank
        jobs = ((candidate._for_simulation(), None, False, False, False) for candidate in candidates)
        results = run_matches(_simulate_match, jobs)
        batch_rank = None
        batch_duration = None
        for candidate, (replay, _) in zip(candidates, results, strict=True):
            rank = _replay_rank(replay)
            duration = replay["duree"]
            if batch_rank is None or rank > batch_rank:
                batch_rank, batch_duration = rank, duration
            if previous_rank is None:
                previous_rank = rank
            if network_rank is None or rank > network_rank:
                network, network_rank, network_duration = candidate, rank, duration
        if endless:
            improved = network_rank > previous_rank
            stalled = 0 if improved else stalled + len(matches)
            if improved or start == 0:
                network.experience = experience
                network.training_matches = previous_matches + stop
                on_record(network, network_rank, network_duration)
            if on_generation is not None:
                on_generation(generation=generation, played=previous_matches + stop,
                              batch=batch_rank, batch_duration=batch_duration, stalled=stalled)
            if stalled == PROGRESS_WINDOW:
                break
        start = stop

    network.experience = experience
    network.training_matches = previous_matches + stop
    return network, network_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Entraîne le réseau et teste de nouvelles poses de dépôt.")
    duration = train.add_mutually_exclusive_group()
    duration.add_argument("--matches", type=int, default=12)
    duration.add_argument("--endless", action="store_true",
                          help=f"Repart de zéro après {PROGRESS_WINDOW} matchs sans amélioration.")
    train.add_argument("--seed", type=int, default=7, help="Graine des tirages (par défaut : %(default)s).")
    train.add_argument("--resume", action="store_true", help="Repart des poids du fichier choisi.")
    train.add_argument("--learning-rate", type=float, default=None, metavar="TAUX",
                       help="Pas d’apprentissage fixe ; par défaut, il diminue avec les matchs déjà joués.")
    commands.add_parser("compare", help="Compare le réseau sauvegardé à l’étape 4 sur la table officielle.")
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.matches < 1:
                parser.error("--matches doit être strictement positif.")
            if args.learning_rate is not None and not args.learning_rate > 0:
                parser.error("--learning-rate doit être strictement positif.")
            incumbent = Network.load(args.model) if args.resume or args.model.exists() else None
            if args.endless:
                _train_endless(args.seed, args.model, incumbent, resume=args.resume,
                               learning_rate=args.learning_rate)
                return
            network, replay = train_matches(args.matches, args.seed, incumbent, resume=args.resume,
                                            learning_rate=args.learning_rate)
            network.save(args.model)
            number, _ = save_replay(replay)
            print(f"Session · {_format_rank(_replay_rank(replay))} en {replay['duree']:.2f} s"
                  f" · replay {number}", flush=True)
            completion_beep()
            return

        network = Network.load(args.model)
        reference = play_match(Simulation(), record_frames=False)
        replay = play_match(Simulation(), policy=MatchPolicy(network))
        score = replay["score"]
        replay.update(strategie="reseau", score_reference=reference["score"])
        number, _ = save_replay(replay)
        print(f"Réseau {score} points en {replay['duree']:.2f} s · "
              f"référence {reference['score']} points en {reference['duree']:.2f} s · replay {number}", flush=True)
        completion_beep()
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
