"""Chargement multi-ventouses et dépose par lots."""

from math import pi, remainder

from pymunk import Vec2d

from ..actions.approche import _pickup_approach, _wall_drop_zones
from ..actions.depose import place_pile
from ..actions.execution import _action_state
from ..actions.prise import _take_pickup
from ..domaine.geometrie import travel_time
from ..domaine.modele import Block
from ..physique.constantes import TURN_SPEED
from ..planification.astar import _clear, _rotated_footprint
from ..planification.base import Approach
from ..regles.phases import BUILD_DEADLINE, FRAME_STEP
from ..regles.score_2027 import construction_score, scoring_zones
from ..terrains.officiel_2027 import TABLE_HEIGHT


def _block_shape(block: Block) -> tuple[Vec2d, ...]:
    """Contour de la pile ramené à son propre repère, sans orientation parasite."""
    center = Vec2d(*block.position)
    return tuple((Vec2d(*corner) - center).rotated(-block.angle) for corner in block.corners)


def _shape_key(shape: tuple[Vec2d, ...]):
    return tuple((round(vertex.x, 6), round(vertex.y, 6)) for vertex in shape)


def _drop_pose_table(shape: tuple[Vec2d, ...], drop_poses):
    """Poses de dépôt qui touchent un rempart, indépendantes du reste du plateau."""
    table = []
    for site, angle in drop_poses:
        relative = _rotated_footprint(shape, angle)
        corners = tuple(tuple(site + corner) for corner in relative)
        zones = _wall_drop_zones(Block(tuple(site), angle, corners))
        if zones[1]:
            table.append((site, angle, zones, relative))
    return tuple(table)


def _play_batches(simulation, record, policy, drop_poses):
    """Charge les ventouses disponibles et dépose les piles une par une."""
    def perform(action, label, **choice):
        def advance(seconds):
            simulation.advance(seconds)
            record(label)

        return action(simulation, **choice, advance=advance, deadline=BUILD_DEADLINE)

    cups = range(1, len(simulation.geometry.mounts) + 1)
    shape_tables = {}
    while simulation.time < BUILD_DEADLINE:
        held = {}
        missed = set()
        while len(held) < len(cups) and simulation.time < BUILD_DEADLINE:
            blocks = simulation.blocks
            carried = set(held.values())
            free_cups = tuple(cup for cup in cups if cup not in held)
            choices = []
            approaches = {}
            state = _action_state(simulation)
            for index, block in enumerate(blocks):
                if index in carried or scoring_zones(block)[1]:
                    continue
                for cup in free_cups:
                    if (index, cup) in missed:
                        continue
                    approach = _pickup_approach(simulation, block, cup, state)
                    if approach is not None:
                        approaches[index, cup] = approach
                        choices.append((approach.duration, index, cup))
            if not choices:
                break
            choices.sort()
            if policy is not None:
                choices = policy.order(simulation, held, choices, pickup=True, can_finish=True, pickup_plans=approaches)
                choice = choices[0]
            else:
                # À un pas de simulation du match près, privilégier le plus petit changement de cap.
                fastest = choices[0][0]
                choice = min((item for item in choices if item[0] <= fastest + FRAME_STEP),
                             key=lambda item: (abs(approaches[item[1:]].approach.turn), item[2], item[0]))
            if choice is None:
                policy.observe(choice)
                if held:
                    break
                record("Fin des constructions · Retour demandé")
                return
            _, index, cup = choice

            if not perform(_take_pickup, f"Prise pile {index + 1} · ventouse {cup}",
                           pile=index, cup=cup, pickup=approaches[index, cup]):
                missed.add((index, cup))
                continue
            if policy is not None:
                policy.observe(choice)
            held[cup] = index
            missed.clear()
            record(f"Pile {index + 1} tenue · {len(held)} pile(s) en charge")
        if not held:
            break

        while held and simulation.time < BUILD_DEADLINE:
            blocks = simulation.blocks
            sites = tuple(scoring_zones(block) for index, block in enumerate(blocks) if index not in held.values())
            score = construction_score(sites)
            start, _, obstacles = _action_state(simulation)
            placements = []
            for cup, index in held.items():
                block = blocks[index]
                center = Vec2d(*block.position)
                footprint = tuple(Vec2d(*corner) - center for corner in block.corners)
                shape = _block_shape(block)
                key = _shape_key(shape)
                if key not in shape_tables:
                    shape_tables[key] = _drop_pose_table(shape, drop_poses)
                for site, angle, zones, relative in shape_tables[key]:
                    if not _clear(site, site, (relative,), obstacles):
                        continue
                    axis_turn = remainder(angle - block.angle, 2 * pi)
                    gain = construction_score((*sites, zones)) - score
                    # À gain égal, éviter de remplir d'abord le passage central des remparts.
                    central_penalty = max(0, 650 - abs(site.y - TABLE_HEIGHT / 2)) / 400
                    # Le rectangle garde la même pose lorsque le châssis l'aborde dans l'autre sens.
                    for offset in ((0,) if policy is None else (0, pi)):
                        turn = axis_turn if not offset else remainder(angle + offset - block.angle, 2 * pi)
                        distance = start.get_distance(site - (center - start).rotated(turn))
                        duration = travel_time(distance) + abs(turn) / TURN_SPEED + central_penalty
                        placements.append((gain, duration, cup, site, angle + offset))
            placements.sort(key=lambda item: (-item[0], item[1]))
            if policy is not None and placements:
                placements = policy.order(simulation, held, placements, pickup=False)
            for choice in placements:
                _, _, cup, site, angle = choice
                index = held[cup]

                def advance_deposit(seconds):
                    simulation.advance(seconds)
                    action = "Recul après dépôt" if simulation.held_pile(cup) is None else "Transport"
                    record(f"{action} pile {index + 1} · ventouse {cup}")

                def record_deposit():
                    record(f"Dépôt pile {index + 1} · {len(held) - 1} pile(s) en charge")

                angles = (angle, angle + pi) if policy is None else (angle,)
                if any(place_pile(simulation, cup, site.x, site.y, drop_angle, advance=advance_deposit,
                                  deadline=BUILD_DEADLINE, on_release=record_deposit)
                       for drop_angle in angles if simulation.time < BUILD_DEADLINE):
                    if policy is not None:
                        policy.observe(choice)
                    held.pop(cup)
                    break
                if simulation.time >= BUILD_DEADLINE:
                    break
            else:
                return


