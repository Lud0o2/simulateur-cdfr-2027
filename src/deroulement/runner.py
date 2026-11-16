"""Déroulement d'un match : construction, retour et replay."""

from math import remainder, pi

from ..actions.retour import _return_home
from ..physique.moteur import Simulation
from ..regles.phases import FRAME_STEP, MATCH_SECONDS
from ..replay.recorder import _frame
from ..strategies.lots import _play_batches
from ..strategies.poses import (
    ALIGNED_ANGLES, ALTERNATE_DROPS, CENTERED_DROPS, DROP_ANGLES, _shifted_drops,
)
from ..strategies.reference import _play_reference


def play_match(simulation: Simulation, *, batches=False, policy=None, record_frames=True,
               full_match=False, drop_poses=None, planner=None):
    batches = batches or policy is not None
    frames = []
    peak = (0, -round(simulation.time, 6))

    def record(label, bonus=None):
        nonlocal peak
        points = None
        if full_match:
            points = simulation.score + (bonus or 0)
            peak = max(peak, (points, -round(simulation.time, 6)))
        if not record_frames:
            return
        if not full_match and frames and simulation.time == 0 and bonus is None:
            return
        frame = _frame(simulation, label, score=points)
        if not full_match:
            frame["score"] += bonus or 0
        if frames and frames[-1]["t"] == frame["t"]:
            if (batches and bonus is None and frames[-1]["score"] == frame["score"]
                    and frames[-1]["action"].startswith(("Pile ", "Dépôt "))):
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
        if drop_poses is None:
            preferred = _shifted_drops(offsets, angles) if offsets else CENTERED_DROPS
            drop_poses = tuple(dict.fromkeys((*preferred, *ALTERNATE_DROPS)))
        _play_batches(simulation, record, policy, drop_poses, planner=planner)
    else:
        _play_reference(simulation, record)

    _return_home(simulation, record, planner=planner)
    bonus = simulation.return_score
    if full_match and simulation.time < MATCH_SECONDS:
        # Le retour est acquis quand le robot termine à l'arrêt et y reste jusqu'au buzzer.
        idle = f"À l’arrêt · Retour +{bonus}" if bonus else "À l’arrêt"
        record(idle, bonus)
        while not simulation.settled and simulation.time < MATCH_SECONDS:
            simulation.advance(min(FRAME_STEP, MATCH_SECONDS - simulation.time))
            record(idle, bonus)
        simulation.advance(MATCH_SECONDS - simulation.time)
    record(f"Fin du match · Retour +{bonus}" if bonus else "Fin du match", bonus)
    replay = {"obstacles": simulation.obstacles, "frames": frames,
              "score": simulation.score + bonus, "score_retour": bonus, "duree": round(simulation.time, 6)}
    if offsets and any(dx or dy for dx, dy in offsets):
        replay["decalages_depot_mm"] = offsets
    if any(angles):
        replay["rotations_depot_rad"] = angles
    if full_match:
        replay.update(score_max=peak[0], temps_score_max=-peak[1])
    return replay


