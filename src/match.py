"""Match automatique sans fenêtre et lecture des essais enregistrés."""

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import json
from math import ceil, isfinite, pi, remainder
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from pymunk import Vec2d

from .simulateur import (
    SPEED, TABLE_HEIGHT, TABLE_WIDTH, TOWER_CENTERS, TURN_SPEED, WALL_ZONES,
    Block, Simulation, construction_score, polygon_contact, rectangle_corners, scoring_zones,
)


MATCH_SECONDS = 100
FRAME_STEP = 0.05
REPLAY_DIR = Path(__file__).resolve().parents[1] / "replays"


def _drop_sites():
    for a, b, c, d in (tuple(Vec2d(*corner) for corner in zone) for zone in WALL_ZONES):
        center = (a + c) / 2
        for length in (0, -0.4, 0.4):
            for width in (0, -0.6, 0.6):
                yield center + (b - a) * length + (d - a) * width
    yield from (Vec2d(*center) for center in TOWER_CENTERS)


DROP_SITES = tuple(_drop_sites())
# Points de passage dans les espaces libres entre les rangées de piles.
WAYPOINTS = tuple(Vec2d(x, y) for x in (300, 600, 950, 1200, 1450, 1800, 2250, 2700)
                  for y in (260, 500, 750, 1000, 1250, 1500, 1740))
OPEN_SPACE = Vec2d(1450, 1000)


@dataclass(frozen=True)
class Plan:
    pile: int
    cup: int
    angle: float
    pickup: tuple[Vec2d, ...]
    drop: tuple[Vec2d, ...]
    retreat: tuple[Vec2d, ...]


def _clear(start, end, footprints, obstacles):
    delta = end - start
    min_x, max_x = sorted((start.x, end.x))
    min_y, max_y = sorted((start.y, end.y))
    for footprint in footprints:
        left, top, right, bottom = _bounds(footprint)
        if not (0 <= end.x + left and end.x + right <= TABLE_WIDTH
                and 0 <= end.y + top and end.y + bottom <= TABLE_HEIGHT):
            return False
        sweep_left, sweep_right = min_x + left, max_x + right
        sweep_top, sweep_bottom = min_y + top, max_y + bottom
        for corners, (x1, y1, x2, y2) in obstacles:
            if sweep_left >= x2 or sweep_right <= x1 or sweep_top >= y2 or sweep_bottom <= y1:
                continue
            if polygon_contact(start, delta, corners, footprint) < 1:
                return False
    return True


def _bounds(corners):
    xs, ys = zip(*corners)
    return min(xs), min(ys), max(xs), max(ys)


def _route(start, end, footprints, obstacles):
    direct = start.get_distance(end)
    if _clear(start, end, footprints, obstacles):
        return (end,), direct
    points = (Vec2d(start.x, end.y), Vec2d(end.x, start.y), *WAYPOINTS)
    detours = ((start.get_distance(point) + point.get_distance(end), point) for point in points)
    for distance, point in sorted(detours, key=lambda detour: detour[0]):
        if _clear(start, point, footprints, obstacles) and _clear(point, end, footprints, obstacles):
            return (point, end), distance
    return (), 0


def choose_plan(simulation: Simulation, unavailable: set[int]) -> Plan | None:
    """Classe les prises suivies d'un dépôt par gain de score par seconde."""
    blocks = simulation.blocks
    sites = tuple(map(scoring_zones, blocks))
    score = construction_score(sites)
    start = Vec2d(*simulation.position)
    fixed = tuple((corners, _bounds(corners)) for corners in map(rectangle_corners, simulation.obstacles))
    piles = tuple((block.corners, _bounds(block.corners)) for block in blocks)
    turns = {}
    candidates = []
    # shortcut: prises centrées et un détour au plus, à étendre pour une table plus encombrée.
    for index, block in enumerate(blocks):
        if index in unavailable or sites[index][1]:
            continue
        center = Vec2d(*block.position)
        relative = tuple(Vec2d(*corner) - center for corner in block.corners)
        obstacles = (*fixed, *piles[:index], *piles[index + 1:])
        deposits = []
        for site in DROP_SITES:
            corners = tuple(tuple(site + corner) for corner in relative)
            zones = scoring_zones(Block(tuple(site), block.angle, corners))
            if zones[1] and _clear(site, site, (relative,), obstacles):
                gain = construction_score((*sites, zones)) - score
                deposits.append((gain, site, (corners, _bounds(corners))))

        if not deposits:
            continue
        pickup_obstacles = (*obstacles, piles[index])
        for side in (-1, 1):
            normal = Vec2d(0, side).rotated(block.angle)
            half_width = max(corner.dot(normal) for corner in relative)
            for cup, (mount, facing) in enumerate(simulation.geometry.mounts, 1):
                heading = (-normal).angle - facing.angle
                angle = remainder(heading - simulation.heading, 2 * pi)
                if angle not in turns:
                    turns[angle] = simulation.can_turn_freely(angle)
                if not turns[angle]:
                    continue
                footprint = tuple(Vec2d(*corner).rotated(heading) for corner in simulation.geometry.contour)
                pickup = center + normal * (half_width + 2) - Vec2d(*mount).rotated(heading)
                if not _clear(pickup, pickup, (footprint,), pickup_obstacles):
                    continue
                approach, distance = _route(start, pickup, (footprint,), pickup_obstacles)
                if not approach:
                    continue
                approach_time = (ceil(abs(angle) / TURN_SPEED / FRAME_STEP) + len(approach)) * FRAME_STEP
                approach_time += distance / SPEED
                held = tuple(corner + center - pickup for corner in relative)
                for gain, site, placed in deposits:
                    drop = site + pickup - center
                    duration = (approach_time + 3 * FRAME_STEP
                                + (drop.get_distance(pickup) + drop.get_distance(OPEN_SPACE)) / SPEED)
                    if simulation.time + duration <= MATCH_SECONDS:
                        candidates.append((gain / duration, gain, duration, index, cup, angle,
                                           approach, drop, footprint, held, obstacles, placed))

    best, best_rate = None, 0
    for rate, gain, duration, index, cup, angle, approach, drop, footprint, held, obstacles, placed in (
        sorted(candidates, key=lambda item: -item[0])
    ):
        if rate <= best_rate:
            break
        if not _clear(drop, drop, (footprint, held), obstacles):
            continue
        retreat, back_distance = _route(drop, OPEN_SPACE, (footprint,), (*obstacles, placed))
        if not retreat:
            continue
        transport, distance = _route(approach[-1], drop, (footprint, held), obstacles)
        if not transport:
            continue
        duration += (distance - approach[-1].get_distance(drop)
                     + back_distance - drop.get_distance(OPEN_SPACE)) / SPEED
        duration += (len(transport) + len(retreat) - 2) * FRAME_STEP
        if simulation.time + duration <= MATCH_SECONDS and gain / duration > best_rate:
            best = Plan(index, cup, angle, approach, transport, retreat)
            best_rate = gain / duration
    return best


def _frame(simulation, label):
    def point(coords):
        return [round(value, 3) for value in coords]

    movement = simulation.movement
    return {
        "t": round(simulation.time, 6),
        "position": point(simulation.position),
        "heading": round(simulation.heading, 6),
        "robot": [point(corner) for corner in simulation.footprint],
        "piles": [[point(corner) for corner in block.corners] for block in simulation.blocks],
        "cups": [[point(cup.position), cup.enabled] for cup in simulation.suction_cups],
        "target": point(movement.end) if simulation.moving and not movement.turn else None,
        "blocked": bool(movement and movement.blocked),
        "score": simulation.score,
        "action": label,
    }


def play_match(simulation: Simulation):
    frames = []
    unavailable: set[int] = set()

    def record(label):
        if frames and simulation.time == 0:
            return
        frame = _frame(simulation, label)
        if frames and frames[-1]["t"] == frame["t"]:
            frames[-1] = frame
        else:
            frames.append(frame)

    def advance(label):
        record(label)
        simulation.advance(min(FRAME_STEP, MATCH_SECONDS - simulation.time))

    def move(target, label):
        if simulation.time >= MATCH_SECONDS:
            return False
        simulation.move_to(*target)
        while simulation.moving and simulation.time < MATCH_SECONDS:
            advance(label)
        return not simulation.moving and Vec2d(*simulation.position).get_distance(target) < 0.01

    record("Départ")
    while simulation.time < MATCH_SECONDS:
        plan = choose_plan(simulation, unavailable)
        if plan is None:
            break
        name = f"pile {plan.pile + 1}"
        steps = ceil(abs(plan.angle) / TURN_SPEED / FRAME_STEP)
        for _ in range(steps):
            if simulation.time >= MATCH_SECONDS:
                break
            simulation.turn_by(plan.angle / steps)
            advance(f"Orientation vers la {name}")
        if not all(move(target, f"Approche de la {name}") for target in plan.pickup):
            unavailable.add(plan.pile)
            continue
        if simulation.time >= MATCH_SECONDS:
            break

        simulation.toggle_suction(plan.cup)
        if simulation.held_pile(plan.cup) != plan.pile:
            simulation.toggle_suction(plan.cup)
            unavailable.add(plan.pile)
            advance(f"Prise manquée · {name}")
            continue
        advance(f"Prise de la {name}")
        delivered = all(move(target, f"Transport de la {name}") for target in plan.drop)
        if simulation.time >= MATCH_SECONDS:
            break
        simulation.toggle_suction(plan.cup)
        advance(f"Dépôt de la {name}" if delivered else f"Transport bloqué · {name}")
        if delivered:
            for target in plan.retreat:
                if not move(target, f"Recul après le dépôt · {name}"):
                    break
        else:
            unavailable.add(plan.pile)

    if simulation.time < MATCH_SECONDS:
        record("Aucun dépôt trouvé dans le temps restant")
        simulation.advance(MATCH_SECONDS - simulation.time)
    record("Fin du match")
    return {"obstacles": simulation.obstacles, "frames": frames}


def _trials(directory: Path):
    trials = {}
    for path in directory.glob("essai-*.json"):
        suffix = path.stem.removeprefix("essai-")
        if suffix.isascii() and suffix.isdecimal() and path.is_file():
            trials[int(suffix)] = path
    return trials


def save_replay(replay, directory: Path = REPLAY_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(dir=directory, prefix=".essai-", suffix=".tmp", encoding="utf-8",
                              mode="w", delete=False)
    temporary = Path(file.name)
    try:
        with file:
            json.dump(replay, file, ensure_ascii=False, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        number = max(_trials(directory), default=0) + 1
        while True:
            path = directory / f"essai-{number:04d}.json"
            try:
                # Sous Windows, rename échoue si la destination existe. Sous POSIX, le lien offre cette garantie.
                if os.name == "nt":
                    temporary.rename(path)
                else:
                    path.hardlink_to(temporary)
            except FileExistsError:
                number += 1
                continue
            return number, path
    finally:
        temporary.unlink(missing_ok=True)


class Replay:
    def __init__(self, data):
        def number(value):
            return type(value) in (int, float) and isfinite(value)

        def coordinates(values, count):
            return (isinstance(values, (list, tuple)) and len(values) == count
                    and all(map(number, values)))

        def polygon(values, minimum):
            return (isinstance(values, list) and len(values) >= minimum
                    and all(coordinates(point, 2) for point in values))

        if not isinstance(data, dict) or not isinstance(data.get("frames"), list) or not data["frames"]:
            raise ValueError("le replay doit contenir au moins une image")
        obstacles = data.get("obstacles")
        if not isinstance(obstacles, (list, tuple)) or not all(coordinates(rect, 4) for rect in obstacles):
            raise ValueError("liste d’obstacles manquante ou invalide")
        self.frames = data["frames"]
        self.obstacles = obstacles
        self.times = []
        for index, frame in enumerate(self.frames, 1):
            if not isinstance(frame, dict):
                raise ValueError(f"image {index} invalide")
            for key in ("t", "position", "heading", "robot", "piles", "cups", "target", "blocked", "score", "action"):
                if key not in frame:
                    raise ValueError(f"image {index} : champ « {key} » manquant")

            timestamp = frame["t"]
            if not number(timestamp) or timestamp < 0 or (self.times and timestamp <= self.times[-1]):
                raise ValueError(f"image {index} : horodatage invalide")
            if not (coordinates(frame["position"], 2) and number(frame["heading"])
                    and (frame["target"] is None or coordinates(frame["target"], 2))
                    and polygon(frame["robot"], 3)
                    and isinstance(frame["piles"], list) and all(polygon(pile, 4) for pile in frame["piles"])
                    and isinstance(frame["cups"], list) and all(
                        isinstance(cup, list) and len(cup) == 2 and coordinates(cup[0], 2) and type(cup[1]) is bool
                        for cup in frame["cups"]
                    )
                    and type(frame["blocked"]) is bool and type(frame["score"]) is int
                    and isinstance(frame["action"], str)):
                raise ValueError(f"image {index} : données d’affichage invalides")
            if index > 1 and any(len(frame[key]) != len(self.frames[0][key]) for key in ("piles", "cups")):
                raise ValueError(f"image {index} : nombre de piles ou de ventouses incohérent")
            self.times.append(timestamp)
        self.start = self.time = self.times[0]
        self.duration = self.times[-1]

    @property
    def frame(self):
        return self.frames[max(0, bisect_right(self.times, self.time) - 1)]


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
            number, path = save_replay(replay)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        print(f"Essai {number} · {replay['frames'][-1]['score']} points en {MATCH_SECONDS} s "
              f"(calcul : {elapsed:.2f} s)")
        print(f"Replay : {path}")
        print(f"python -B -m src.match replay {number}")


if __name__ == "__main__":
    main()
