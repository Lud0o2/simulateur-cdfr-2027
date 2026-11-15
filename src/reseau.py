"""Entraînement et évaluation du réseau sur des matchs simulés."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

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
from .reseaux.entrainement import (
    ContexteEntrainement, _train_endless as _train_endless_impl,
    _train_matches as _train_matches_impl, train_matches as _train_matches_run,
)
from .recompenses.construction_temps import ConstructionTemps
from .evaluation.metriques import _format_elapsed, _match_rank, _replay_rank
from .selection.reseau import MatchPolicy
from .strategies.poses import ALIGNED_ANGLES, CENTERED_DROPS, DROP_ANGLES


EXPLORATION_INTERVAL = 8
PROGRESS_WINDOW = 50  # matchs d'entraînement sans nouveau record autonome
MODEL = MODELE
RECOMPENSE = ConstructionTemps()


def _contexte():
    return ContexteEntrainement(
        play_match=play_match,
        simulation=Simulation,
        policy=MatchPolicy,
        network=Network,
        process_pool=ProcessPoolExecutor,
        save_replay=save_replay,
        mutate=_mutate_drop_pose,
        recompense=RECOMPENSE,
        replay_rank=_replay_rank,
        learning_rate=_learning_rate,
        format_elapsed=_format_elapsed,
        progress_window=PROGRESS_WINDOW,
        exploration_interval=EXPLORATION_INTERVAL,
    )


def train_matches(count, seed, incumbent=None, *, resume=False, learning_rate=None):
    return _train_matches_run(_contexte(), count, seed, incumbent, resume=resume,
                              learning_rate=learning_rate)


def _train_endless(seed, model, incumbent, *, resume, learning_rate=None):
    return _train_endless_impl(_contexte(), seed, model, incumbent, resume=resume,
                               learning_rate=learning_rate)


def _train_matches(count, seed, incumbent, resume, workers, run_matches, *, on_record=None,
                   on_generation=None, learning_rate=None):
    return _train_matches_impl(_contexte(), count, seed, incumbent, resume, workers, run_matches,
                               on_record=on_record, on_generation=on_generation,
                               learning_rate=learning_rate)



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
