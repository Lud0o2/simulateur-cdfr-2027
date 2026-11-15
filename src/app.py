"""Point d'entrée unifié du simulateur."""

import argparse
import sys

from .composition.config import charger, defaut
from .composition.defaut import composer


def _match(args):
    from .deroulement.runner import play_match
    from .infra.notificateur import completion_beep
    from .match import save_replay
    from .physique.moteur import Simulation

    composition = composer(charger(args.profil) if args.profil else defaut())
    replay = play_match(Simulation(), policy=composition.selecteur, planner=composition.planificateur)
    number, _ = save_replay(replay)
    print(f"Essai {number} · {replay['score']} points en {replay['duree']:.2f} s simulées", flush=True)
    completion_beep()


def _replay(args):
    from .replay.lecture import Replay
    from .replay.store import _trials
    from .infra.ressources import REPLAY_DIR
    import json

    trials = _trials(REPLAY_DIR)
    number = args.number if args.number is not None else max(trials, default=0)
    if number not in trials:
        raise SystemExit(f"Essai {number} introuvable." if args.number is not None else "Aucun essai enregistré.")
    with trials[number].open(encoding="utf-8") as file:
        replay = Replay(json.load(file))
    from .interface import create_window

    create_window(replay=replay, trial=number).mainloop()


def _ui(_args):
    from .interface import create_window
    from .physique.moteur import Simulation

    create_window(Simulation()).mainloop()


def _delegate_train(rest):
    from . import reseau

    sys.argv = ["reseau", *rest]
    reseau.main()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("match", help="Joue un match avec la composition choisie.")
    match.add_argument("--profil", type=str)
    match.set_defaults(run=_match)
    replay = commands.add_parser("replay", help="Ouvre un replay.")
    replay.add_argument("number", nargs="?", type=int)
    replay.set_defaults(run=_replay)
    ui = commands.add_parser("ui", help="Ouvre le pilotage manuel.")
    ui.set_defaults(run=_ui)
    train = commands.add_parser("train", help="Délègue à src.reseau.")
    train.add_argument("rest", nargs=argparse.REMAINDER)
    train.set_defaults(run=lambda args: _delegate_train(["train", *args.rest]))
    compare = commands.add_parser("compare", help="Délègue à src.reseau compare.")
    compare.set_defaults(run=lambda args: _delegate_train(["compare"]))
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
