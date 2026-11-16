"""Entraîne et évalue les réseaux sur des matchs simulés."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from copy import copy
from functools import partial
from itertools import batched
from math import cos, dist, hypot, pi, remainder, sin
import os
from pathlib import Path
from secrets import randbits
import signal
from tempfile import NamedTemporaryFile

import numpy as np
from pymunk import Vec2d

from .match import (
    ALIGNED_ANGLES, CENTERED_DROPS, DROP_ANGLES, MATCH_SECONDS, REPLAY_DIR, _action_state, _drop_target,
    _pickup_approach, _rotated_footprint, _visibility, _wall_drop_zones,
    completion_beep, play_collection, play_match, save_replay,
)
from .simulateur import Block, START, START_BLOCKS, TABLE_HEIGHT, TABLE_WIDTH, WALL_ZONES, Simulation, free_wall_slots, scoring_zones


LAYERS = (97, 32, 16, 1)
TABLE_DIAGONAL = hypot(TABLE_WIDTH, TABLE_HEIGHT)
ACTION_DISTANCE = 84
MODEL = REPLAY_DIR.parent / "reseau.npz"
COLLECTION_MODEL = REPLAY_DIR.parent / "reseau-collecte.npz"
COLLECTION_LAYERS = (11, 8, 1)
PICKUP_LIMIT = 8
BATTLE_SIZE = 50
MUTANTS_PER_NETWORK = 10
MUTATION_SPREADS = (0.04, 0.12, 0.30)
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


class Network:
    def __init__(self, rng, layers=LAYERS):
        self.weights = [rng.normal(0, (2 / size) ** 0.5, (size, width)).astype(np.float32)
                        for size, width in zip(layers, layers[1:])]
        self.biases = [np.zeros(width, dtype=np.float32) for width in layers[1:]]
        self.drop_offsets = CENTERED_OFFSETS
        self.drop_angles = ALIGNED_ANGLES

    def predict(self, inputs):
        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            inputs = np.maximum(inputs @ weights + biases, 0)
        return (inputs @ self.weights[-1] + self.biases[-1])[:, 0]

    def save(self, path):
        file = NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
        temporary = Path(file.name)
        try:
            with file:
                np.savez(file, *self.weights, *self.biases, drop_offsets=np.asarray(self.drop_offsets),
                         drop_angles=np.asarray(self.drop_angles))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, *, layers=LAYERS):
        construction = layers == LAYERS
        with np.load(path, allow_pickle=False) as data:
            layer_count = len(layers) - 1
            parameters = [data[f"arr_{index}"] for index in range(2 * layer_count)]
            if construction:
                hidden = parameters[1]
                if hidden.ndim != 2 or not all(hidden.shape):
                    raise ValueError("Paramètres du réseau invalides.")
                layers = (LAYERS[0], *hidden.shape, LAYERS[-1])
            shapes = list(zip(layers, layers[1:])) + [(width,) for width in layers[1:]]
            presence_bias = 0
            for index, (loaded, shape) in enumerate(zip(parameters, shapes, strict=True)):
                legacy_input = construction and index == 0 and loaded.shape == (layers[0] - 1, layers[1])
                if ((loaded.shape != shape and not legacy_input) or loaded.dtype != np.float32
                        or not np.all(np.isfinite(loaded))):
                    raise ValueError("Paramètres du réseau invalides.")
                if legacy_input:
                    # Les anciennes lignes de présence sont reprises par le biais.
                    presence_bias = loaded[11:77:7].sum(axis=0)
                    loaded = np.insert(loaded, ACTION_DISTANCE, 0, axis=0)
                    loaded[11:77:7] = 0
                    loaded[78] /= 2
                    parameters[index] = loaded
            network = cls(np.random.default_rng(0), layers)
            network.weights, network.biases = parameters[:layer_count], parameters[layer_count:]
            network.biases[0] += presence_bias
            if not np.all(np.isfinite(network.biases[0])):
                raise ValueError("Paramètres du réseau invalides.")
            if "drop_offsets" in data:
                network.drop_offsets = _read_drop_offsets(data["drop_offsets"].tolist())
            if "drop_angles" in data:
                network.drop_angles = _read_drop_angles(data["drop_angles"].tolist())
        return network


class MatchPolicy:
    def __init__(self, network):
        self.network = network
        self.drop_offsets = getattr(network, "drop_offsets", CENTERED_OFFSETS)
        self.drop_angles = getattr(network, "drop_angles", ALIGNED_ANGLES)

    def order(self, simulation, held, choices, *, pickup, can_finish=False, pickup_plans=None):
        # shortcut: le réseau classe les huit prises les plus rapides, à élargir si elles limitent la stratégie.
        limit = min(PICKUP_LIMIT, len(choices)) if pickup else len(choices)
        head = choices[:limit]
        if can_finish:
            head.append(None)
        inputs = features(simulation, held, head, pickup=pickup, pickup_plans=pickup_plans)
        order = np.argsort(-self.network.predict(inputs), kind="stable")
        return [head[index] for index in order] + choices[limit:]

    def observe(self, _choice):
        pass


def _replay_rank(replay):
    return replay["score"], -replay["duree"]


def _elimination_score(rank):
    points, negative_duration = rank
    return points + negative_duration / 2


def _validated_replay(network, rank, seed):
    replay = play_match(Simulation(), policy=MatchPolicy(network))
    if _replay_rank(replay) != rank:
        raise ValueError("Le replay du réseau ne reproduit pas son résultat de validation.")
    replay.update(strategie="validation_reseau", graine=seed)
    return replay


def _validation_rank(network):
    replay = play_match(Simulation(), policy=MatchPolicy(network), record_frames=False)
    return _replay_rank(replay)


def _mutate_weights(network, rng, spread):
    mutant = copy(network)
    parameters = [values + rng.normal(0, spread, values.shape).astype(values.dtype)
                  for values in (*network.weights, *network.biases)]
    count = len(network.weights)
    mutant.weights, mutant.biases = parameters[:count], parameters[count:]
    return mutant


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


def _mutate_contender(network, rng):
    if rng.random() < 0.25:
        return _mutate_drop_pose(network, rng)
    return _mutate_weights(network, rng, rng.choice(MUTATION_SPREADS))


def _training_workers(count):
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return min(count, available, 61 if os.name == "nt" else available)


def _battle_royale(rng, run_matches, on_result):
    def evaluate(candidates):
        for network, rank in zip(candidates, run_matches(_validation_rank, candidates), strict=True):
            on_result(network, rank)
            yield rank, network

    entrants = [Network(rng) for _ in range(BATTLE_SIZE)]
    population = list(evaluate(entrants))
    for round_number in range(1, BATTLE_SIZE):
        mutants = [_mutate_contender(network, rng)
                   for _, network in population for _ in range(MUTANTS_PER_NETWORK)]
        groups = batched(evaluate(mutants), MUTANTS_PER_NETWORK)
        population = [max((parent, *variants), key=lambda trial: trial[0])
                      for parent, variants in zip(population, groups, strict=True)]
        population.sort(key=lambda trial: (_elimination_score(trial[0]), trial[0]), reverse=True)
        eliminated, _ = population.pop()
        score, duration = eliminated
        best, _ = population[0]
        print(f"Manche {round_number}/{BATTLE_SIZE - 1} · éliminé {score} pts en {-duration:.2f} s"
              f" · points − temps/2 = {_elimination_score(eliminated):.2f}"
              f" · {len(population)} en lice · tête {best[0]} pts en {-best[1]:.2f} s", flush=True)
    return population[0][0]


def _train_battle_royale(seed, model, *, endless):
    rng = np.random.default_rng(seed)
    workers = _training_workers(BATTLE_SIZE)
    print(f"Battle royale · {BATTLE_SIZE} réseaux · {MUTANTS_PER_NETWORK} mutants par réseau et par manche"
          f" · {workers} processus · graine {seed}", flush=True)
    saved = Network.load(model) if model.exists() else None
    champion = (_validation_rank(saved), saved) if saved is not None else None
    replay_rank = number = None

    def save_champion():
        nonlocal saved, replay_rank, number
        rank, network = champion
        if rank != replay_rank:
            replay = _validated_replay(network, rank, seed)
            if network is not saved:
                network.save(model)
                saved = network
            number, _ = save_replay(replay, descriptive=True)
            replay_rank = rank

    def consider(network, rank):
        nonlocal champion
        if champion is None or rank > champion[0]:
            champion = rank, network
            save_champion()
            print(f"Record · {rank[0]} points en {-rank[1]:.2f} s · replay {number}", flush=True)

    pool = (ProcessPoolExecutor(max_workers=workers, initializer=signal.signal,
                                initargs=(signal.SIGINT, signal.SIG_IGN)) if workers > 1 else None)
    try:
        tournament = 1
        while True:
            print(f"Tournoi {tournament} · évaluation des {BATTLE_SIZE} réseaux de départ", flush=True)
            score, duration = _battle_royale(rng, pool.map if pool else map, consider)
            print(f"Vainqueur du tournoi {tournament} · {score} pts en {-duration:.2f} s", flush=True)
            if not endless:
                break
            tournament += 1
    except KeyboardInterrupt:
        print("Tournoi interrompu.", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)

    if champion is not None:
        save_champion()
        (score, duration), _ = champion
        print(f"Session · {score} points en {-duration:.2f} s · replay {number}", flush=True)
        completion_beep()


def _pick_collection(network, simulation, held, actions):
    carried = set(held.values())
    stocks = [(index, block.position) for index, block in enumerate(simulation.blocks) if index not in carried]
    mounts = simulation.geometry.mounts
    rows = []
    for (distance, duration, index, cup, pickup), pair in actions:
        neighbor = min((dist(pickup.end, position) for other, position in stocks if other != index),
                       default=0)
        _, facing = mounts[cup - 1]
        rows.append((distance / TABLE_DIAGONAL, duration / 5, pickup.approach.turn / pi,
                     dist(pickup.end, START) / TABLE_DIAGONAL, neighbor / TABLE_DIAGONAL,
                     *facing, len(held) / len(mounts), len(stocks) / len(START_BLOCKS),
                     (MATCH_SECONDS - simulation.time) / MATCH_SECONDS, pair is not None))
    return actions[int(np.argmax(network.predict(np.asarray(rows, dtype=np.float32))))]


def _simulate_collection(network, *, record_frames=False):
    return play_collection(Simulation(), policy=partial(_pick_collection, network), record_frames=record_frames)


def _train_collection(count, seed, incumbent=None):
    rng = np.random.default_rng(seed)
    workers = _training_workers(count)
    population = []
    print(f"Collecte · {workers} processus pour {count} matchs · graine {seed}", flush=True)
    with (ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool:
        run = pool.map if pool else map
        for start in range(0, count, 8):
            size = min(8, count - start)
            previous = population[0][0] if population else None
            candidates = [Network(rng, COLLECTION_LAYERS) for _ in range(size if not population else min(2, size))]
            if start == 0 and incumbent is not None:
                candidates[0] = incumbent
            mutations = ((parent, spread) for spread in MUTATION_SPREADS for _, parent in population)
            for parent, spread in mutations:
                if len(candidates) == size:
                    break
                candidates.append(_mutate_weights(parent, rng, spread))
            population.extend((_replay_rank(replay), candidate)
                              for candidate, replay in zip(candidates, run(_simulate_collection, candidates), strict=True))
            population = sorted(population, key=lambda trial: trial[0], reverse=True)[:2]
            rank = population[0][0]
            if previous is None or rank > previous:
                print(f"Matchs {start + size}/{count} · record {rank[0]} piles en {-rank[1]:.2f} s", flush=True)
    rank, network = population[0]
    return network, rank


def _run_collection(args, model):
    incumbent = Network.load(model, layers=COLLECTION_LAYERS) if args.resume or args.jouer or model.exists() else None
    if args.jouer:
        replay = _simulate_collection(incumbent, record_frames=True)
    else:
        seed = args.seed if args.seed is not None else randbits(32)
        network, rank = _train_collection(args.matches, seed, incumbent if args.resume else None)
        if incumbent is not None and not args.resume:
            previous = _replay_rank(_simulate_collection(incumbent))
            if previous >= rank:
                network, rank = incumbent, previous
        replay = _simulate_collection(network, record_frames=True)
        if _replay_rank(replay) != rank:
            raise ValueError("Le replay de collecte ne reproduit pas son résultat de validation.")
        network.save(model)
        replay.update(graine=seed, matchs_entrainement=args.matches)
    replay["politique"] = "reseau_collecte"
    number, _ = save_replay(replay, descriptive=True)
    print(f"Collecte · {replay['score']}/{replay['total_piles']} piles en {replay['duree']:.2f} s · replay {number}",
          flush=True)
    completion_beep()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help=f"Lance un tournoi à {BATTLE_SIZE} réseaux.")
    train.add_argument("--endless", action="store_true", help="Enchaîne les tournois jusqu’à Ctrl+C.")
    train.add_argument("--seed", type=int, help="Graine d’entraînement, tirée au hasard par défaut.")
    collection = commands.add_parser("collecte", help="Entraîne ou joue le réseau de collecte seule.")
    mode = collection.add_mutually_exclusive_group()
    mode.add_argument("--matches", type=int, default=100, help="Nombre de matchs d'entraînement.")
    mode.add_argument("--jouer", action="store_true", help="Joue avec les poids de collecte enregistrés.")
    collection.add_argument("--seed", type=int, help="Graine d'entraînement, tirée au hasard par défaut.")
    collection.add_argument("--resume", action="store_true", help="Repart des poids de collecte enregistrés.")
    commands.add_parser("compare", help="Compare le réseau sauvegardé à l’étape 4 sur la table officielle.")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "collecte":
            if args.matches < 1:
                parser.error("--matches doit être strictement positif.")
            if args.resume and args.jouer:
                parser.error("--resume s'applique à l'entraînement.")
            _run_collection(args, args.model or COLLECTION_MODEL)
            return
        model = args.model or MODEL
        if args.command == "train":
            seed = args.seed if args.seed is not None else randbits(32)
            _train_battle_royale(seed, model, endless=args.endless)
            return

        network = Network.load(model)
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
