"""Déroulement d'un match : construction, retour et replay."""

from math import remainder, pi

from ..actions.retour import _return_home
from ..physique.moteur import Simulation
from ..replay.recorder import _frame
from ..strategies.lots import _play_batches
from ..strategies.poses import (
    ALIGNED_ANGLES, ALTERNATE_DROPS, CENTERED_DROPS, DROP_ANGLES, _shifted_drops,
)
from ..strategies.reference import _play_reference


def play_match(simulation: Simulation, *, batches=False, policy=None, record_frames=True):
    batches = batches or policy is not None
    frames = []

    def record(label, bonus=None):
        if not record_frames:
            return
        if frames and simulation.time == 0 and bonus is None:
            return
        frame = _frame(simulation, label)
        frame["score"] += bonus or 0
        if frames and frames[-1]["t"] == frame["t"]:
            if batches and bonus is None and frames[-1]["action"].startswith(("Pile ", "Dépôt ")):
                return
            frames[-1] = frame
        else:
            frames.append(frame)

    record("Départ")
    offsets = getattr(policy, "drop_offsets", None)
    angles = getattr(policy, "drop_angles", ALIGNED_ANGLES)
    if batches:
        angles = tuple(min(DROP_ANGLES, key=lambda candidate: abs(remainder(candidate - angle, pi)))
                       for angle in angles)
        drop_poses = _shifted_drops(offsets, angles) if offsets else CENTERED_DROPS
        _play_batches(simulation, record, policy, tuple(dict.fromkeys((*drop_poses, *ALTERNATE_DROPS))))
    else:
        _play_reference(simulation, record)

    _return_home(simulation, record)
    bonus = simulation.return_score
    record(f"Fin du match · Retour +{bonus}" if bonus else "Fin du match", bonus)
    replay = {"obstacles": simulation.obstacles, "frames": frames,
              "score": simulation.score + bonus, "score_retour": bonus, "duree": round(simulation.time, 6)}
    if offsets and any(dx or dy for dx, dy in offsets):
        replay["decalages_depot_mm"] = offsets
    if any(angles):
        replay["rotations_depot_rad"] = angles
    return replay


