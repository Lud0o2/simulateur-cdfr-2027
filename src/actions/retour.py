"""Retour dans la salle du trône."""

from math import pi, remainder

from pymunk import Vec2d

from ..domaine.geometrie import _polygon_bounds as _bounds
from ..planification.astar import _approach, _segment_time
from ..regles.phases import FRAME_STEP, MATCH_SECONDS
from ..regles.score_2027 import throne_room_score
from ..terrains.officiel_2027 import START, THRONE_ROOM
from .execution import _action_state, _follow
from .approche import _planifier


def _return_home(simulation, record, planner=None):
    deadline = MATCH_SECONDS - FRAME_STEP  # Une arrivée au buzzer doit déjà être à l'arrêt.
    if simulation.time >= deadline:
        return
    if simulation.moving:
        simulation.move_to(*simulation.position)
    current = simulation.return_score
    if current == 10:
        return

    start, footprints, obstacles = _action_state(simulation)
    _, top, right, bottom = THRONE_ROOM
    targets = {10: [], 5: []}
    # shortcut: cinq poses par cap, à élargir si les accès de la salle sont encombrés.
    turns = dict.fromkeys((0, *(remainder(heading - simulation.heading, 2 * pi)
                              for heading in (0, pi / 2, pi, -pi / 2))))
    for turn in turns:
        hull = tuple(Vec2d(*corner).rotated(turn) for corner in footprints[0])
        left_edge, _, right_edge, _ = _bounds(hull)
        for target in (Vec2d(*START), Vec2d(right - right_edge - 10, START[1]),
                       Vec2d(right - left_edge - 20, START[1]), Vec2d(START[0], top + 50),
                       Vec2d(START[0], bottom - 50)):
            score = throne_room_score(target, hull)
            if score > current:
                targets[score].append((target, turn))

    def advance(seconds):
        simulation.advance(seconds)
        record("Retour à la salle du trône")

    for score in (10, 5):
        best = None
        for target, turn in targets[score]:
            direct = _segment_time(start, target)
            if best is not None and direct >= best.duration:
                continue
            route = _planifier(planner, start, target, turn, footprints, obstacles)
            if (route is not None and simulation.time + route.duration <= deadline
                    and (best is None or route.duration < best.duration)):
                best = route
        if best is not None:
            if (_follow(simulation, best, advance=advance, deadline=deadline)
                    and current == 0 and simulation.return_score == 5):
                _return_home(simulation, record, planner=planner)
            return


