"""Boucle d'entraînement avec dépendances injectées."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import os
import signal
from time import monotonic
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ContexteEntrainement:
    play_match: Callable
    simulation: Callable
    policy: Callable
    network: Callable
    process_pool: Callable = ProcessPoolExecutor
    save_replay: Callable | None = None
    mutate: Callable | None = None
    recompense: object | None = None
    replay_rank: Callable | None = None
    learning_rate: Callable | None = None
    format_elapsed: Callable | None = None
    progress_window: int = 50
    exploration_interval: int = 8
    clock: Callable = monotonic


def _training_workers(count):
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return min(count, available, 61 if os.name == "nt" else available)


def _validated_replay(ctx, network, rank, seed):
    replay = ctx.play_match(ctx.simulation(), policy=ctx.policy(network))
    if ctx.replay_rank(replay) != rank:
        raise ValueError("Le replay du réseau ne reproduit pas son résultat de validation.")
    replay.update(strategie="validation_reseau", graine=seed)
    return replay


def _simulate_match(ctx, job):
    network, seed, warmup, resume, random_match = job
    rng = None if seed is None else np.random.default_rng(seed)
    policy = ctx.policy(network, rng=rng, warmup=warmup, resume=resume, random_match=random_match)
    replay = ctx.play_match(ctx.simulation(), policy=policy, record_frames=False)
    if policy.opening and replay["score"] > replay["score_retour"]:
        replay["ouverture"] = tuple((policy.opening + [-1])[:2])
    return replay, policy.history


def train_matches(ctx, count, seed, incumbent=None, *, resume=False, learning_rate=None):
    workers = _training_workers(count)
    print(f"Simulation · {workers} processus pour {count} matchs d’entraînement", flush=True)
    with (ctx.process_pool(max_workers=workers) if workers > 1 else nullcontext()) as pool:
        network, rank = _train_matches(ctx, count, seed, incumbent, resume, workers, pool.map if pool else map,
                                       learning_rate=learning_rate)
    replay = _validated_replay(ctx, network, rank, seed)
    if incumbent is not None and not resume:
        previous, _ = _simulate_match(ctx, (incumbent, None, False, False, False))
        if ctx.replay_rank(previous) >= rank:
            incumbent.experience = network.experience
            incumbent.training_matches = network.training_matches
            network = incumbent
    return network, replay


def _train_endless(ctx, seed, model, incumbent, *, resume, learning_rate=None):
    started = ctx.clock()
    saved_rank = None
    improved_at = None
    session_best = None
    series_best = None
    series_replays = {}
    series = 1
    seeds = np.random.SeedSequence(seed)
    workers = _training_workers(ctx.progress_window)

    def record(network, rank):
        nonlocal saved_rank, improved_at, session_best, series_best
        if series_best is None or rank > series_best[1]:
            series_best = (network, rank)
        if session_best is None or rank > session_best[0]:
            session_best = rank, series
        if saved_rank is None or rank > saved_rank:
            network.save(model)
            saved_rank = rank
            improved_at = ctx.clock()

    def save_series():
        nonlocal series_best
        if series_best is None:
            return
        network, rank = series_best
        replay = _validated_replay(ctx, network, rank, seed)
        replay.update(strategie="entrainement_endless", serie=series)
        number, _ = ctx.save_replay(replay, descriptive=True)
        series_replays[series] = number
        series_best = None
        print(f"Série {series} · {rank[0]} points en {-rank[1]:.2f} s · replay {number}", flush=True)

    def status(*, generation, played, batch, stalled):
        rank = saved_rank if saved_rank is not None else batch
        since = (f"aucune amélioration depuis {ctx.format_elapsed(ctx.clock() - started)}"
                 if improved_at is None else f"amélioration il y a {ctx.format_elapsed(ctx.clock() - improved_at)}")
        print(f"Série {series} · génération {generation} · {played} matchs"
              f" · record {rank[0]} pts en {-rank[1]:.2f} s"
              f" · meilleur du lot {batch[0]} pts en {-batch[1]:.2f} s"
              f" · {since} · stagnation {stalled}/{ctx.progress_window}", flush=True)

    try:
        if incumbent is not None:
            replay, _ = _simulate_match(ctx, (incumbent, None, False, False, False))
            saved_rank = ctx.replay_rank(replay)
        with (ctx.process_pool(max_workers=workers, initializer=signal.signal,
                                  initargs=(signal.SIGINT, signal.SIG_IGN)) if workers > 1 else nullcontext()) as pool:
            run_matches = pool.map if pool else map
            starting_network = incumbent if resume else None
            while True:
                print(f"Série {series} · graine {seed} · "
                      f"{'reprise du modèle' if starting_network is not None else 'nouveau réseau aléatoire'}",
                      flush=True)
                _train_matches(ctx, None, seed, starting_network, starting_network is not None,
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


def _train_matches(ctx, count, seed, incumbent, resume, workers, run_matches, *, on_record=None,
                   on_generation=None, learning_rate=None):
    endless = count is None
    rng = np.random.default_rng(seed)
    mutation_rng = np.random.default_rng(seed)
    network = incumbent if resume else ctx.network(rng)
    network_rank = None
    stalled = 0
    generation = 0
    experience = incumbent.experience.copy() if incumbent is not None else {}
    previous_matches = incumbent.training_matches if incumbent is not None else 0
    played = previous_matches if resume else 0
    learning_rate = ctx.learning_rate(played) if learning_rate is None else learning_rate
    warmup_matches = max(1, (ctx.progress_window if endless else count) // 3)
    seeds = np.random.SeedSequence(seed, n_children_spawned=played)
    examples = deque(maxlen=512)
    start = 0
    while endless or start < count:
        generation += 1
        remaining = ctx.progress_window - stalled if endless else count - start
        size = min(workers, remaining)
        stop = start + size
        # La mémoire sert à l'apprentissage dans le processus principal.
        playing = network._for_simulation()
        jobs = [(playing, match_seed, not resume and episode < warmup_matches, resume,
                 (played + episode + 1) % ctx.exploration_interval == 0)
                for episode, match_seed in enumerate(seeds.spawn(size), start)]
        # Tous les matchs du lot doivent partir des mêmes poids avant leur mise à jour.
        matches = list(run_matches(partial(_simulate_match, ctx), jobs))

        candidates = [ctx.mutate(network, mutation_rng) for _ in matches]
        remembered = tuple(experience.values())
        ranked = sorted((match for match in matches if match[1]), key=lambda match: ctx.replay_rank(match[0]), reverse=True)
        for index, (replay, history) in enumerate(ranked):
            # Le temps départage les scores égaux avec un bonus de 0 à 0,5 point.
            current = [(row, ctx.recompense.valeur_match(score, replay)) for row, score in history]
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
                rank = ctx.replay_rank(replay)
                if opening not in experience or rank > experience[opening][0]:
                    rows, rewards = zip(*current)
                    experience[opening] = rank, np.column_stack((np.asarray(rows), rewards)).astype(np.float32)

        if network_rank is None:
            candidates.insert(0, network)
        previous_rank = network_rank
        jobs = ((candidate._for_simulation(), None, False, False, False) for candidate in candidates)
        results = run_matches(partial(_simulate_match, ctx), jobs)
        batch_rank = None
        for candidate, (replay, _) in zip(candidates, results, strict=True):
            rank = ctx.replay_rank(replay)
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
            if stalled == ctx.progress_window:
                break
        start = stop

    network.experience = experience
    network.training_matches = previous_matches + stop
    return network, network_rank
