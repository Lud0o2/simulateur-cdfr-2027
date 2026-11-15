"""Compare plusieurs politiques sur un scénario."""

from ..deroulement.runner import play_match
from ..physique.moteur import Simulation
from .metriques import _replay_rank


def evaluer(politiques, scenario=None):
    """Renvoie un dictionnaire nom -> rang du replay."""
    resultats = {}
    for nom, politique in politiques.items():
        simulation = Simulation(
            obstacles=() if scenario is None else scenario.obstacles,
            blocks=None if scenario is None else scenario.piles_initiales,
        )
        replay = play_match(simulation, policy=politique, record_frames=False)
        resultats[nom] = _replay_rank(replay)
    return resultats
