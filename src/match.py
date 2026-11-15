"""Façade de compatibilité de l'ancien module `match`.

Le code réel est réparti entre `planification`, `actions`, `strategies`,
`deroulement` et `replay`. Les tests et les anciens imports continuent de
fonctionner via cette façade.
"""

import argparse
from time import perf_counter

from .actions.approche import _drop_approach, _drop_target, _pickup_approach, _wall_drop_zones
from .actions.depose import place_pile
from .actions.execution import _action_state, _follow, _stroke_end
from .actions.params import DROP_RETREAT, PICKUP_GAP, PICKUP_STANDOFF, PICKUP_STROKE
from .actions.prise import _take_pickup, take_pile
from .actions.retour import _return_home
from .deroulement.runner import play_match
from .infra.ressources import REPLAY_DIR
from .physique.moteur import Simulation
from .planification.astar import OPEN_SPACE, ROUTE_MARGIN, _approach, _clear, _route, _rotated_footprint, _visibility
from .planification.base import Approach, Pickup, Plan
from .regles.phases import BUILD_DEADLINE, FRAME_STEP, MATCH_SECONDS
from .replay.io import save_replay as _save_replay
from .replay.lecture import Replay
from .replay.recorder import _frame
from .replay.store import _trials
from .strategies.lots import _block_shape, _drop_pose_table, _play_batches, _shape_key
from .strategies.poses import (
    ALIGNED_ANGLES, ALTERNATE_DROPS, CENTERED_DROPS, DROP_ANGLES, DROP_SITES, _shifted_drops,
)
from .strategies.reference import _play_reference, choose_plan
from .infra.notificateur import completion_beep
import json


def save_replay(replay, directory=None, *, descriptive=False):
    return _save_replay(
        replay, REPLAY_DIR if directory is None else directory,
        descriptive=descriptive, trials=_trials,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    player = commands.add_parser("replay", help="Ouvre le dernier essai ou le numéro indiqué.")
    player.add_argument("number", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "replay":
        trials = _trials(REPLAY_DIR)
        number = args.number if args.number is not None else max(trials, default=0)
        if number not in trials:
            parser.error(f"Essai {number} introuvable." if args.number is not None else "Aucun essai enregistré.")
        from .interface import create_window

        try:
            with trials[number].open(encoding="utf-8") as file:
                replay = Replay(json.load(file))
        except (OSError, ValueError, OverflowError) as error:
            parser.error(f"Impossible de lire l'essai {number} ({error}).")
        create_window(replay=replay, trial=number).mainloop()
    else:
        try:
            start = perf_counter()
            replay = play_match(Simulation())
            elapsed = perf_counter() - start
            number, _ = save_replay(replay)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        print(f"Essai {number} · {replay['score']} points en {replay['duree']:.2f} s simulées "
              f"(calcul : {elapsed:.2f} s)", flush=True)
        completion_beep()


if __name__ == "__main__":
    main()
