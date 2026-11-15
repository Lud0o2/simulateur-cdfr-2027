"""Entraînement et évaluation du réseau sur des matchs simulés."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import os
from pathlib import Path
import signal
from time import monotonic

import numpy as np

from .infra.notificateur import completion_beep
from .infra.ressources import MODELE, REPLAY_DIR
from .match import play_match, save_replay
from .physique.moteur import Simulation
from .regles.phases import MATCH_SECONDS
from .reseaux.base import ACTION_DISTANCE, LAYERS, PICKUP_LIMIT, TABLE_DIAGONAL
from .reseaux.caracteristiques import features
from .reseaux.mlp import (
    ADAM_BETAS, ADAM_EPSILON, BATCH_SIZE, CENTERED_OFFSETS, DROP_OFFSET_LIMIT, EPOCHS,
    LEARNING_RATE, LEARNING_RATE_FLOOR, LEARNING_RATE_HALVING, MAX_STEPS, MOMENT_NAMES,
    Network, _learning_rate,
)
from .reseaux.mutation import _mutate_drop_pose
from .recompenses.construction_temps import ConstructionTemps
from .evaluation.metriques import _format_elapsed, _match_rank, _replay_rank
from .selection.reseau import MatchPolicy
from .strategies.poses import ALIGNED_ANGLES, CENTERED_DROPS, DROP_ANGLES


EXPLORATION_INTERVAL = 8
PROGRESS_WINDOW = 50  # matchs d'entraînement sans nouveau record autonome
MODEL = MODELE
RECOMPENSE = ConstructionTemps()


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
    improved_at = None
    session_best = None
    series_best = None
    series_replays = {}
    series = 1
    seeds = np.random.SeedSequence(seed)
    workers = _training_workers(PROGRESS_WINDOW)

    def record(network, rank):
        nonlocal saved_rank, improved_at, session_best, series_best
        if series_best is None or rank > series_best[1]:
            series_best = (network, rank)
        if session_best is None or rank > session_best[0]:
            session_best = rank, series
        if saved_rank is None or rank > saved_rank:
            network.save(model)
            saved_rank = rank
            improved_at = monotonic()

    def save_series():
        nonlocal series_best
        if series_best is None:
            return
        network, rank = series_best
        replay = _validated_replay(network, rank, seed)
        replay.update(strategie="entrainement_endless", serie=series)
        number, _ = save_replay(replay, descriptive=True)
        series_replays[series] = number
        series_best = None
        print(f"Série {series} · {rank[0]} points en {-rank[1]:.2f} s · replay {number}", flush=True)

    def status(*, generation, played, batch, stalled):
        rank = saved_rank if saved_rank is not None else batch
        since = (f"aucune amélioration depuis {_format_elapsed(monotonic() - started)}"
                 if improved_at is None else f"amélioration il y a {_format_elapsed(monotonic() - improved_at)}")
        print(f"Série {series} · génération {generation} · {played} matchs"
              f" · record {rank[0]} pts en {-rank[1]:.2f} s"
              f" · meilleur du lot {batch[0]} pts en {-batch[1]:.2f} s"
              f" · {since} · stagnation {stalled}/{PROGRESS_WINDOW}", flush=True)

    try:
        if incumbent is not None:
            replay, _ = _simulate_match((incumbent, None, False, False, False))
            saved_rank = _replay_rank(replay)
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
        rank, best_series = session_best
        print(f"Session · {rank[0]} points en {-rank[1]:.2f} s"
              f" · série {best_series} · replay {series_replays[best_series]}", flush=True)


def _train_matches(count, seed, incumbent, resume, workers, run_matches, *, on_record=None,
                   on_generation=None, learning_rate=None):
    endless = count is None
    rng = np.random.default_rng(seed)
    mutation_rng = np.random.default_rng(seed)
    network = incumbent if resume else Network(rng)
    network_rank = None
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
            current = [(row, RECOMPENSE.valeur_match(score, replay)) for row, score in history]
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
        for candidate, (replay, _) in zip(candidates, results, strict=True):
            rank = _replay_rank(replay)
            if batch_rank is None or rank > batch_rank:
                batch_rank = rank
            if previous_rank is None:
                previous_rank = rank
            if network_rank is None or rank > network_rank:
                network, network_rank = candidate, rank
        if endless:
            improved = network_rank > previous_rank
            stalled = 0 if improved else stalled + len(matches)
            if improved or start == 0:
                network.experience = experience
                network.training_matches = previous_matches + stop
                on_record(network, network_rank)
            if on_generation is not None:
                on_generation(generation=generation, played=previous_matches + stop,
                              batch=batch_rank, stalled=stalled)
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
            print(f"Session · {replay['score']} points en {replay['duree']:.2f} s · replay {number}", flush=True)
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
