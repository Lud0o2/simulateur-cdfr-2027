"""Neuroévolution d'un réseau acyclique qui classe les actions du simulateur."""

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import blake2b
import json
from math import cos, degrees, isfinite, radians, remainder, sin, tau
import os
from pathlib import Path
from random import Random
from secrets import randbits
from statistics import mean, median
from tempfile import NamedTemporaryFile, mkdtemp
from threading import Event
from time import perf_counter

import numpy as np
from pymunk import Vec2d

from .match import CENTERED_DROPS, MATCH_SECONDS, REPLAY_DIR, Replay, play_match, save_replay
from .reseau import _training_workers
from .simulateur import TABLE_HEIGHT, TABLE_WIDTH, Simulation, _finite_number, scoring_zones


RUN_DIR = REPLAY_DIR.parent / "neat"
CHAMPION_FILE = "champion.json"
SCENARIO = "CDFR 2027 · table officielle fixe"
INPUTS = ("Prise", "Fin de série", "Durée estimée", "Gain de points", "Charge", "Temps restant",
          "Stock restant", "Score actuel", "Robot X", "Robot Y", "Cap X", "Cap Y", "Cible X", "Cible Y",
          "Rotation X", "Rotation Y", "Ventouse X", "Ventouse Y", "Sortie côté stocks", "Charge si fin")
PRIOR = (1.8, -1.3, -2.0, 2.2, *([0.0] * (len(INPUTS) - 6)), 1.65, 2.4)
N_INPUTS = len(INPUTS)
MAX_HIDDEN = 12
MAX_LINKS = 120
WEIGHT_LIMIT = 32
WEIGHT_RESET_RATE = 0.01
TARGET_SCORE = 65
STRATEGY_ARCHIVE_LIMIT = 128
DESCENDANT_SIGMA_FACTOR = 1.5

# shortcut: un échantillon fixe de poses réalisables, à élargir si la capacité des remparts limite le score.
DROP_VARIANTS = (
    (0, 0, 120, 30), (1, 0, 120, 30), (1, -120, -160, 30), (2, 0, 120, -80),
    (2, 0, -120, 0), (3, -120, -160, -50), (3, 0, -120, -50), (3, 0, 120, 20),
    (4, 0, 120, -70), (4, 0, -120, -10),
)
DROP_POSES = (
    *CENTERED_DROPS,
    *((site + Vec2d(0, offset).rotated(axis), axis)
      for site, axis in CENTERED_DROPS for offset in (-120, 120)),
    *((CENTERED_DROPS[index][0] + Vec2d(x, y).rotated(CENTERED_DROPS[index][1]),
       CENTERED_DROPS[index][1] + radians(angle)) for index, x, y, angle in DROP_VARIANTS),
)
DROP_POSES = tuple(dict.fromkeys(DROP_POSES))


@dataclass(frozen=True)
class Settings:
    population: int = 200
    generations: int = 0
    seed: int = 0
    sigma: float = 0.28
    add_node: float = 0.40
    add_link: float = 0.50

    def __post_init__(self):
        if (self.population < 2 or self.generations < 0 or not isfinite(self.sigma)
                or self.sigma <= 0 or not 0 <= self.add_node <= 1 or not 0 <= self.add_link <= 1):
            raise ValueError("Population ≥ 2, générations ≥ 0, sigma > 0, probabilités entre 0 et 1.")


@dataclass(frozen=True)
class Genome:
    # Les entrées sont suivies des neurones cachés en ordre topologique, puis de la sortie.
    biases: tuple[float, ...]
    links: tuple[tuple[int, int, float], ...]

    @property
    def output(self):
        return N_INPUTS + len(self.biases) - 1

    def data(self):
        return dict(biases=list(self.biases), links=list(map(list, self.links)))

    @classmethod
    def read(cls, data):
        biases, links = data["biases"], data["links"]
        if (not isinstance(biases, list) or not 1 <= len(biases) <= MAX_HIDDEN + 1
                or not all(map(_finite_number, biases))
                or not isinstance(links, list) or not 1 <= len(links) <= MAX_LINKS):
            raise ValueError("Dimensions ou biais du génome invalides.")
        output = N_INPUTS + len(biases) - 1
        endpoints = set()
        for link in links:
            if not isinstance(link, list) or len(link) != 3:
                raise ValueError("Connexion du génome invalide.")
            source, target, weight = link
            if (type(source) is not int or type(target) is not int or not 0 <= source < target <= output
                    or target < N_INPUTS or not _finite_number(weight) or (source, target) in endpoints):
                raise ValueError("Connexion du génome invalide.")
            endpoints.add((source, target))
        return cls(tuple(biases), tuple(map(tuple, links)))

    def weights(self):
        matrix = np.zeros((len(self.biases), self.output), dtype=np.float64)
        for source, target, weight in self.links:
            matrix[target - N_INPUTS, source] = weight
        return matrix


def initial_genome(rng, spread):
    links = tuple((index, N_INPUTS, weight + rng.gauss(0, spread) if spread else weight)
                  for index, weight in enumerate(PRIOR))
    return Genome((0.0,), links)


def mutate(genome, rng, settings, spread):
    def perturb(value):
        return max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, value + rng.gauss(0, spread)))

    biases = list(genome.biases)
    links = list(genome.links)
    for index in rng.sample(range(len(links)), rng.randint(1, min(4, len(links)))):
        source, target, weight = links[index]
        links[index] = source, target, perturb(weight)
    index = rng.randrange(max(1, len(biases) - 1))
    biases[index] = perturb(biases[index])
    for index, (source, target, _) in enumerate(links):
        if rng.random() < WEIGHT_RESET_RATE:
            links[index] = source, target, rng.uniform(-1.0, 1.0)

    if len(biases) <= MAX_HIDDEN and len(links) < MAX_LINKS and rng.random() < settings.add_node:
        source, target, weight = links.pop(rng.randrange(len(links)))
        links = [(a + (a >= target), b + (b >= target), w) for a, b, w in links]
        biases.insert(target - N_INPUTS, 0.0)
        # Pour de petites activations, tanh(x / 2) × 2 conserve presque la réponse de la connexion.
        links.extend(((source, target, 0.5), (target, target + 1, 2 * weight)))

    if len(links) < MAX_LINKS and rng.random() < settings.add_link:
        connected = {(a, b) for a, b, _ in links}
        pairs = [(a, b) for b in range(N_INPUTS, N_INPUTS + len(biases))
                 for a in range(b) if (a, b) not in connected]
        if pairs:
            source, target = rng.choice(pairs)
            links.append((source, target, rng.gauss(0, spread)))
    return Genome(tuple(biases), tuple(links))


def observations(simulation, held, choices, *, pickup, can_finish=False, pickup_plans=None):
    values = np.zeros((len(choices) + can_finish, N_INPUTS), dtype=np.float64)
    x, y = simulation.position
    heading = simulation.heading
    mounts = simulation.geometry.mounts
    blocks = simulation.blocks
    carried = set(held.values())
    free = sum(index not in carried and not scoring_zones(block)[1] for index, block in enumerate(blocks))
    load = len(held) / len(mounts)
    # À la prise, les candidats possèdent déjà un trajet calculé. Au dépôt, la durée est une estimation.
    values[:, 4:12] = (load, 1 - simulation.time / MATCH_SECONDS, free / 10, simulation.score_for_blocks(blocks) / 65,
                       x / TABLE_WIDTH, y / TABLE_HEIGHT, cos(heading), sin(heading))
    for row, choice in zip(values, choices):
        exit_side = 0
        if pickup:
            duration, index, cup = choice
            position = blocks[index].position
            rotation = pickup_plans[index, cup].approach.turn
            gain = 0
        else:
            gain, duration, cup, position, angle = choice
            block = blocks[held[cup]]
            rotation = remainder(angle - block.angle, tau)
            # Le châssis du côté des stocks évite de fermer le passage derrière lui.
            bx, by = block.position
            if free:
                exit_side = ((by - y) * sin(rotation) - (bx - x) * cos(rotation)) / 250
        mount, _ = mounts[cup - 1]
        row[:4] = (pickup, 0, min(duration / 4, 4), gain / 8)
        row[12:] = (position[0] / TABLE_WIDTH, position[1] / TABLE_HEIGHT, cos(rotation), sin(rotation),
                    mount[0] / 250, mount[1] / 250, exit_side, 0)
    if can_finish:
        values[-1, 1] = 1
        values[-1, -1] = load
    return values


class NeatPolicy:
    def __init__(self, genome):
        self.matrix = genome.weights()
        self.biases = genome.biases
        self.actions = []

    def order(self, simulation, held, choices, *, pickup, can_finish=False, pickup_plans=None):
        inputs = observations(simulation, held, choices, pickup=pickup,
                              can_finish=can_finish, pickup_plans=pickup_plans)
        values = np.empty((len(inputs), N_INPUTS + len(self.biases)), dtype=np.float64)
        values[:, :N_INPUTS] = inputs
        for index, (weights, bias) in enumerate(zip(self.matrix, self.biases), N_INPUTS):
            activation = values[:, :index] @ weights[:index] + bias
            values[:, index] = np.tanh(activation) if index + 1 < values.shape[1] else activation
        candidates = [*choices, None] if can_finish else choices
        return [candidates[index] for index in np.argsort(-values[:, -1], kind="stable")]

    def observe(self, choice):
        if choice is None:
            self.actions.append(("finish",))
        elif len(choice) == 3:
            _, pile, cup = choice
            self.actions.append(("take", pile, cup))
        else:
            _, _, cup, site, angle = choice
            self.actions.append(("drop", cup, round(site.x), round(site.y), round(degrees(angle)) % 180))


def fitness(match):
    return match["score_max"], -match["temps_score_max"]


def evaluate(genome, record_frames=False):
    policy = NeatPolicy(genome)
    match = play_match(Simulation(), policy=policy, drop_poses=DROP_POSES,
                       full_match=True, record_frames=record_frames)
    if record_frames:
        return match
    rank = fitness(match)
    strategy = blake2b(repr(policy.actions).encode(), digest_size=12).hexdigest() if rank[0] >= TARGET_SCORE else None
    return rank, strategy


@dataclass(frozen=True)
class Individual:
    id: int
    generation: int
    genome: Genome
    rank: tuple[int, float] | None = None
    strategy: str | None = None

    def selection_key(self):
        return self.rank[1], -self.id


def remember_strategies(archive, qualified):
    for individual in sorted(qualified, key=lambda item: item.id):
        strategy = individual.strategy
        previous = archive.get(strategy)
        if previous is None and len(archive) == STRATEGY_ARCHIVE_LIMIT:
            del archive[next(iter(archive))]
        if previous is None or individual.selection_key() > previous.selection_key():
            archive[strategy] = individual


def offspring(population, rng, settings, generation, first_id, archive):
    fresh = settings.population // 10
    inherited = settings.population - fresh
    archived = inherited // 4 if archive else 0
    parents = rng.choices(archive, k=archived)
    parents.extend(max(rng.choices(population, k=3), key=lambda item: item.rank)
                   for _ in range(inherited - archived))
    genomes = [parent.genome for parent in parents]
    genomes.extend(initial_genome(rng, 0.8) for _ in range(fresh))
    spread = settings.sigma * DESCENDANT_SIGMA_FACTOR
    return [Individual(first_id + i, generation, mutate(genome, rng, settings, spread * rng.choice((0.5, 1, 2))))
            for i, genome in enumerate(genomes)]


def write_champion(directory, individual, match, settings, initial=None):
    if fitness(match) != individual.rank or max((frame["score"], -frame["t"]) for frame in match["frames"]) != individual.rank:
        raise ValueError("Le replay ne reproduit pas le score ou le temps mesuré.")
    data = {"version": 1, "scenario": SCENARIO, "settings": asdict(settings),
            "id": individual.id, "generation": individual.generation,
            "score": individual.rank[0], "time": -individual.rank[1],
            "genome": individual.genome.data(), "replay": match}
    if initial is not None:
        data["initial_genome"] = initial.data()
    file = NamedTemporaryFile(dir=directory, prefix=".champion-", suffix=".tmp",
                              mode="w", encoding="utf-8", delete=False)
    temporary = Path(file.name)
    try:
        with file:
            json.dump(data, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(directory / CHAMPION_FILE)
    finally:
        temporary.unlink(missing_ok=True)
    return data


def read_champion(directory):
    data = json.loads((directory / CHAMPION_FILE).read_text(encoding="utf-8"))
    if data["version"] != 1 or data["scenario"] != SCENARIO:
        raise ValueError("Version ou scénario du champion incompatible.")
    if any(type(data[key]) is not int or data[key] < 1 for key in ("id", "generation")):
        raise ValueError("Identifiant ou génération du champion invalide.")
    genome = Genome.read(data["genome"])
    replay = Replay(data["replay"])
    if (replay.start != 0 or replay.duration != MATCH_SECONDS
            or (data["score"], -data["time"]) != fitness(data["replay"])
            or (data["score"], -data["time"]) != max((frame["score"], -frame["t"]) for frame in replay.frames)):
        raise ValueError("Score, temps ou durée du replay incohérent.")
    return data, genome, replay


def new_directory(path):
    if path is not None:
        path.mkdir(parents=True)
        return path
    RUN_DIR.mkdir(exist_ok=True)
    return Path(mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=RUN_DIR))


def latest_directory():
    for path in sorted(RUN_DIR.glob(f"*/{CHAMPION_FILE}"), reverse=True):
        if json.loads(path.read_text(encoding="utf-8"))["scenario"] == SCENARIO:
            return path.parent
    raise FileNotFoundError("Aucun champion NEAT en solo. Lance d'abord l'entraînement.")


def train(settings, directory, publish, *, initial=None, pause=None, stop=None):
    pause = pause if pause is not None else Event()
    stop = stop if stop is not None else Event()
    rng = Random(settings.seed)
    genomes = [initial or initial_genome(rng, 0)]
    for index in range(1, settings.population):
        if initial is not None and index % 4:
            genomes.append(mutate(initial, rng, settings, settings.sigma * rng.choice((0.25, 0.5, 1, 2))))
        else:
            genomes.append(initial_genome(rng, rng.uniform(0.18, 1.15)))
    population = [Individual(index, 1, genome) for index, genome in enumerate(genomes, 1)]
    workers = _training_workers(max(1, settings.population - 1))
    workers = min(workers, max(1, (os.cpu_count() or 1) - 1))
    started = perf_counter()
    champion = saved = None
    recording = None
    simulations = completed = 0
    generation = 1
    archive = {}
    results = []

    def report(status="Entraînement", **event):
        elapsed = perf_counter() - started
        publish(dict(status=status, generation=generation, completed=completed, population=settings.population,
                     evaluated=len(results), simulations=simulations, elapsed=elapsed,
                     rate=simulations / max(elapsed, 0.001), **event))

    def save_recording(recording):
        future, individual = recording
        match = future.result()
        data = write_champion(directory, individual, match, settings, initial)
        number, _ = save_replay(match, REPLAY_DIR, descriptive=True)
        report(champion=data, log=f"Champion #{individual.id} sauvegardé · replay {number}")
        return individual

    report(log=f"Graine {settings.seed} · seuil {TARGET_SCORE} pts · {workers} processus")
    if initial is not None:
        report(log="Population initialisée autour du génome importé")
    with ProcessPoolExecutor(max_workers=workers) as pool, (directory / "generations.jsonl").open(
        "a", encoding="utf-8",
    ) as history:
        while not stop.is_set() and (settings.generations == 0 or generation <= settings.generations):
            results = []
            qualified = []
            waiting = deque(population)
            tasks = {}
            report(log=f"Génération {generation}")
            while tasks or (waiting and not stop.is_set()):
                if recording is None and champion != saved:
                    recording = (pool.submit(evaluate, champion.genome, True), champion)
                if not (pause.is_set() or stop.is_set()):
                    while waiting and len(tasks) + bool(recording) < workers:
                        individual = waiting.popleft()
                        tasks[pool.submit(evaluate, individual.genome)] = individual

                futures = tuple(tasks) + ((recording[0],) if recording else ())
                if not futures:
                    report("Pause")
                    stop.wait(0.25)
                    continue

                if pause.is_set():
                    report("Pause demandée")
                ready, _ = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                for task in sorted(ready & tasks.keys(), key=lambda item: tasks[item].id):
                    rank, strategy = task.result()
                    individual = replace(tasks.pop(task), rank=rank, strategy=strategy)
                    results.append(individual)
                    simulations += 1
                    message = {}
                    score, time = individual.rank
                    if score >= TARGET_SCORE:
                        qualified.append(individual)
                        if champion is None or individual.selection_key() > champion.selection_key():
                            if champion is None or time > champion.rank[1]:
                                change = "Objectif atteint" if champion is None else "Temps amélioré"
                                message["log"] = f"{change} · {score} pts en {-time:.2f} s · réseau #{individual.id}"
                            champion = individual
                    report("Pause demandée" if pause.is_set() else "Entraînement", **message)
                if recording and recording[0] in ready:
                    saved = save_recording(recording)
                    recording = None

            if len(results) != settings.population:
                break
            remember_strategies(archive, qualified)
            scores = [item.rank[0] for item in results]
            leader = (max(qualified, key=Individual.selection_key) if qualified
                      else max(results, key=lambda item: (item.rank, -item.id)))
            stats = dict(generation=generation, id=leader.id, best=leader.rank[0], time=-leader.rank[1],
                         qualified=len(qualified), archived_strategies=len(archive),
                         mean=mean(scores), median=median(scores), distribution=sorted(Counter(scores).items()),
                         simulations=simulations, elapsed=perf_counter() - started, sigma=settings.sigma * DESCENDANT_SIGMA_FACTOR)
            history.write(json.dumps(stats, separators=(",", ":")) + "\n")
            history.flush()
            completed += 1
            report(stats=stats, log=(f"Génération {generation} · {len(qualified)}/{settings.population} qualifiés"
                                    f" · meilleur {leader.rank[0]} pts en {-leader.rank[1]:.2f} s"
                                    f" · {len(archive)} stratégies en mémoire"))
            if stop.is_set() or completed == settings.generations:
                break
            generation += 1
            population = offspring(sorted(results, key=lambda item: item.id), rng, settings, generation,
                                   (generation - 1) * settings.population + 1,
                                   [archive[key] for key in sorted(archive)])
        if recording is not None:
            saved = save_recording(recording)
        if champion != saved:
            save_recording((pool.submit(evaluate, champion.genome, True), champion))
    if stop.is_set():
        detail = f"champion #{champion.id} sauvegardé" if champion is not None else "aucun réseau qualifié"
        report("Terminé", log=f"Entraînement arrêté · {detail}")
    else:
        report("Terminé", log="Entraînement terminé")
    return champion


def train_background(settings, directory, initial, events, pause, stop):
    started = perf_counter()

    def publish(event):
        if "champion" in event:
            event["replay"] = Replay(event["champion"]["replay"])
        events.put(event)

    try:
        train(settings, new_directory(directory), publish, initial=initial, pause=pause, stop=stop)
    except Exception as exc:
        publish(dict(status="Erreur", elapsed=perf_counter() - started, log=str(exc)))


def main():
    parser = argparse.ArgumentParser(description="Neuroévolution du robot CDFR")
    parser.add_argument("mode", nargs="?", choices=("live", "train", "replay", "verify"), default="live")
    parser.add_argument("--directory", type=Path, help="dossier d'une session NEAT")
    parser.add_argument("--population", type=int, default=Settings.population)
    parser.add_argument("--generations", type=int, help="0 pour continuer jusqu'à l'arrêt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sigma", type=float, default=0.28)
    parser.add_argument("--add-node", type=float, default=Settings.add_node)
    parser.add_argument("--add-link", type=float, default=Settings.add_link)
    parser.add_argument("--from-genome", type=Path, help="génome JSON ou champion.json pour initialiser la population")
    args = parser.parse_args()
    try:
        if args.mode in {"replay", "verify"}:
            directory = args.directory or latest_directory()
            data, genome, replay = read_champion(directory)
            if args.mode == "verify":
                match = evaluate(genome, record_frames=True)
                if fitness(match) != (data["score"], -data["time"]) or match["frames"] != replay.frames:
                    raise ValueError("Le génome rechargé ne reproduit pas le match.")
                print(f"Champion #{data['id']} · {data['score']} points en {data['time']:.2f} s · replay reproduit")
            else:
                from .interface_neat import run_window
                run_window(saved=(data, replay), directory=directory)
            return
        generations = args.generations if args.generations is not None else (0 if args.mode == "live" else 10)
        seed = args.seed if args.seed is not None else randbits(32)
        settings = Settings(args.population, generations, seed, args.sigma, args.add_node, args.add_link)
        initial = None
        if args.from_genome is not None:
            data = json.loads(args.from_genome.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Génome JSON invalide.")
            initial = Genome.read(data.get("genome", data))
        if args.mode == "live":
            from .interface_neat import run_window
            run_window(settings=settings, directory=args.directory, initial=initial)
        else:
            directory = new_directory(args.directory)
            print(f"NEAT · {directory}", flush=True)

            def log(event):
                if "log" in event:
                    print(event["log"], flush=True)

            train(settings, directory, log, initial=initial)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"NEAT · {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "NEAT · Entraînement arrêté\n")


if __name__ == "__main__":
    main()
