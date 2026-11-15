import json
import unittest
from functools import partial
from itertools import groupby
from math import cos, dist, pi, radians, remainder, sin, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import mock_open, patch

import numpy as np
from pymunk import Vec2d

from src.match import (
    ALIGNED_ANGLES, ALTERNATE_DROPS, CENTERED_DROPS, MATCH_SECONDS, ROUTE_MARGIN, Replay, _action_state, _approach, _clear,
    _drop_approach, _follow, _pickup_approach, _route, place_pile, play_match, save_replay, take_pile,
)
from src.reseau import (
    ACTION_DISTANCE, CENTERED_OFFSETS, DROP_OFFSET_LIMIT, EXPLORATION_INTERVAL, LAYERS, PROGRESS_WINDOW, TABLE_DIAGONAL,
    MatchPolicy, Network, _mutate_drop_pose, _replay_rank, features, train_matches,
)
from src.simulateur import (
    START, START_BLOCKS, TOWER_CENTERS, TURN_STEP, RobotGeometry, Simulation, load_robot,
    rectangle_corners, scoring_zones, travel_time,
)


# La géométrie de référence garde les scénarios reproductibles après édition du TOML.
SIDE = 380
FRONT = SIDE * sqrt(3) / 6
FOOTPRINT = ((-2 * FRONT, 0), (FRONT, -SIDE / 2), (FRONT, SIDE / 2))
reference_simulation = partial(Simulation, robot=RobotGeometry(FOOTPRINT, (
    ((FRONT, 0), Vec2d(1, 0)),
    ((-FRONT / 2, SIDE / 4), Vec2d(1, 0).rotated(radians(120))),
    ((-FRONT / 2, -SIDE / 4), Vec2d(1, 0).rotated(radians(-120))),
)))
two_pile_simulation = partial(
    Simulation,
    blocks=((1445, 440, 1555, 760), (1445, 1040, 1555, 1360)),
    robot=RobotGeometry(FOOTPRINT, (((FRONT, -100), Vec2d(1, 0)), ((FRONT, 100), Vec2d(1, 0)))),
)


class TrainingReplayTest(unittest.TestCase):
    def test_random_match_explores_past_the_eight_fastest_pickups(self):
        policy = MatchPolicy(object(), rng=np.random.default_rng(7), random_match=True)
        choices = [(1 + pile, pile, 1) for pile in range(10)]
        simulation = Simulation()
        selected = set()
        for _ in range(40):
            ordered = policy.order(simulation, {}, choices, pickup=True, can_finish=True)
            self.assertCountEqual(ordered, choices)
            selected.add(ordered[0][1])
        self.assertIn(9, selected)

    def test_keeps_and_reuses_an_opening_below_the_record_after_reloading(self):
        incumbent = Network(np.random.default_rng(7))
        exploration_modes = []
        remembered = np.full(LAYERS[0], 2, dtype=np.float32)

        def play(_simulation, *, policy, record_frames=True):
            if policy.rng is None:
                return {"score": 50, "duree": 20}
            if not record_frames:
                exploration_modes.append(policy.random_match)
            if policy.random_match:
                policy.history.append((remembered, 0))
                return {"score": 42, "duree": 25, "ouverture": (4, 1)}
            policy.history.append((np.zeros(LAYERS[0], dtype=np.float32), 0))
            return {"score": 48, "duree": 21}

        with (patch("src.reseau.play_match", side_effect=play),
              patch("src.reseau.os.process_cpu_count", return_value=1, create=True),
              patch.object(Network, "train") as train,
              TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory):
            best, _ = train_matches(8, 7, incumbent, resume=True)
            self.assertIs(best, incumbent)
            self.assertEqual(exploration_modes, [False] * 7 + [True])
            path = Path(directory) / "reseau.npz"
            best.save(path)
            loaded = Network.load(path)
            self.assertEqual(loaded.experience[4, 1][0], (42, -25))
            self.assertEqual(loaded.training_matches, 8)
            train.reset_mock()
            train_matches(1, 7, loaded, resume=True)
            inputs, targets, _ = train.call_args.args
            np.testing.assert_array_equal(inputs[-1], remembered)
            self.assertAlmostEqual(float(targets[-1]), (42 + 0.5 * (1 - 25 / MATCH_SECONDS)) / 8)

    def test_short_sessions_continue_the_random_match_schedule(self):
        network = Network(np.random.default_rng(7))
        total = EXPLORATION_INTERVAL * (PROGRESS_WINDOW // EXPLORATION_INTERVAL + 1)
        network.training_matches = total - 1

        def play(_simulation, *, policy, record_frames=True):
            if policy.rng is not None:
                self.assertTrue(policy.random_match)
            return {"score": 50, "duree": 20}

        with patch("src.reseau.play_match", side_effect=play):
            train_matches(1, 7, network, resume=True)
        self.assertEqual(network.training_matches, total)

    def test_endless_resets_from_scratch_and_keeps_only_the_session_best(self):
        incumbent = Network(np.random.default_rng(7))
        incumbent.training_matches = 49
        incumbent.biases[-1][0] = 99
        incumbent.drop_angles = (radians(10), *incumbent.drop_angles[1:])
        incumbent.experience[0, 1] = (50, -20), np.full((1, LAYERS[0] + 1), 42, dtype=np.float32)
        scores = {99: (50, 20), 1: (51, 25), 0: (40, 30), 2: (50, 15)}
        played = [0, 0]
        series = 0

        def play(_simulation, *, policy, record_frames=True):
            nonlocal series
            if policy.rng is None:
                score, duration = scores[policy.network.biases[-1][0]]
                return {"score": score, "duree": duration}
            if played[1] == 53:
                raise KeyboardInterrupt
            if played[0] == 56:
                series = 1
            played[series] += 1
            self.assertLessEqual(played[series], 56)
            if series == 1 and played[1] == 1:
                self.assertFalse(policy.resume)
                self.assertEqual(policy.network.training_matches, 0)
                self.assertEqual(policy.network.biases[-1][0], 0)
                self.assertNotEqual(policy.drop_angles, incumbent.drop_angles)
                self.assertFalse(np.array_equal(policy.network.weights[0], incumbent.weights[0]))
            policy.history.append((np.zeros(LAYERS[0]), 0))
            return {"score": 64, "duree": 12}

        def update(network, inputs, _targets, _rng, *, learning_rate):
            if series == 0 and played[0] >= 6:
                network.biases[-1][0] = 1
            if series == 1:
                self.assertFalse((inputs == 42).any())
                network.biases[-1][0] = 2

        def save_session(replay, *, descriptive):
            return save_replay(replay, directory=Path(directory), descriptive=descriptive)

        from src.reseau import main

        with (TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory,
              patch("src.reseau.play_match", side_effect=play), patch.object(Network, "train", update),
              patch("src.reseau.PROGRESS_WINDOW", 50),
              patch("src.reseau.os.process_cpu_count", return_value=3, create=True),
              patch("src.reseau.ProcessPoolExecutor") as pool, patch("builtins.print") as output,
              patch("src.reseau._mutate_drop_pose", wraps=_mutate_drop_pose) as mutate,
              patch("src.reseau.save_replay", side_effect=save_session)):
            pool.return_value.__enter__.return_value.map = map
            model = Path(directory) / "reseau.npz"
            incumbent.save(model)
            with patch("sys.argv", ["reseau", "--model", str(model), "train", "--endless", "--resume"]):
                main()
            self.assertEqual(Network.load(model).biases[-1][0], 1)
            paths = sorted(Path(directory).glob("essai-*.json"))
            replays = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

        self.assertEqual(played, [56, 53])
        self.assertEqual(mutate.call_count, sum(played))
        self.assertEqual([path.name for path in paths], ["essai-0001-51pts-25.00s.json"])
        self.assertEqual(replays[0]["serie"], 1)
        self.assertEqual(_replay_rank(replays[0]), (51, -25))
        output.assert_called_once_with("Session · 51 points en 25.00 s · replay 1", flush=True)

    def test_deposit_mutations_are_kept_only_when_the_autonomous_match_improves(self):
        incumbent = Network(np.random.default_rng(0))
        first = _mutate_drop_pose(incumbent, np.random.default_rng(4))
        second = _mutate_drop_pose(first, np.random.default_rng(0))
        self.assertNotEqual(first.drop_offsets, incumbent.drop_offsets)
        self.assertNotEqual(second.drop_angles, first.drop_angles)
        ranks = {(incumbent.drop_offsets, incumbent.drop_angles): (50, 22.7),
                 (first.drop_offsets, first.drop_angles): (50, 22.0),
                 (second.drop_offsets, second.drop_angles): (49, 21.0)}

        def play(_simulation, *, policy, record_frames=True):
            score, duration = ranks[policy.drop_offsets, policy.drop_angles]
            return {"score": score, "duree": duration}

        with (patch("src.reseau.play_match", side_effect=play),
              patch("src.reseau._mutate_drop_pose", side_effect=(first, second)) as mutate,
              patch("src.reseau.os.process_cpu_count", return_value=1, create=True)):
            best, winner = train_matches(2, 7, incumbent, resume=True)
        self.assertEqual([call.args[0] for call in mutate.call_args_list], [incumbent, first])
        self.assertIs(best, first)
        self.assertEqual(_replay_rank(winner), (50, -22))
        self.assertEqual(incumbent.drop_offsets, CENTERED_OFFSETS)

    def test_deposit_mutations_can_reach_200_mm_and_keep_a_single_changed_pose(self):
        network = Network(np.random.default_rng(0))
        rng = np.random.default_rng(9)
        farthest = 0
        for _ in range(64):
            mutant = _mutate_drop_pose(network, rng)
            farthest = max(farthest, *(dist((0, 0), offset) for offset in mutant.drop_offsets))
            changes = sum(a != b for a, b in zip(mutant.drop_offsets, network.drop_offsets))
            changes += sum(a != b for a, b in zip(mutant.drop_angles, network.drop_angles))
            self.assertEqual(changes, 1)
            for angle in mutant.drop_angles:
                self.assertAlmostEqual(angle / radians(10), round(angle / radians(10)))
        self.assertEqual(farthest, DROP_OFFSET_LIMIT)

    def test_each_generation_continues_with_the_best_network_on_the_official_table(self):
        incumbent = Network(np.random.default_rng(7))
        incumbent.biases[-1][0] = 0
        official_blocks = Simulation().blocks
        scores = {0: (40, 30), 1: (48, 40), 2: (44, 20), 3: (48, 30), 4: (46, 10), 5: (48, 30), 6: (42, 15)}
        versions = iter(range(1, 7))
        evaluated_weights = []
        training_weights = []
        candidate_parents = []
        targets = []
        recorded_winners = []

        def play(simulation, *, policy, record_frames=True):
            self.assertEqual(simulation.blocks, official_blocks)
            version = int(policy.network.biases[-1][0])
            if policy.rng is not None:
                self.assertFalse(record_frames)
                training_weights.append(version)
                policy.history.append((np.zeros(LAYERS[0]), 0))
                return {"score": 64 if len(training_weights) in (2, 4) else 60,
                        "duree": 13 if len(training_weights) == 4 else 15}
            (recorded_winners if record_frames else evaluated_weights).append(version)
            score, duration = scores[version]
            return {"score": score, "duree": duration}

        def update(network, _rows, rewards, _rng, *, learning_rate):
            targets.extend(rewards)
            candidate_parents.append(int(network.biases[-1][0]))
            network.biases[-1][0] = next(versions)

        with (patch("src.reseau.play_match", side_effect=play), patch.object(Network, "train", update),
              patch("src.reseau.os.process_cpu_count", return_value=2, create=True),
              patch("src.reseau.ProcessPoolExecutor") as pool):
            pool.return_value.__enter__.return_value.map = map
            best, winner = train_matches(6, 7, incumbent, resume=True)
        pool.assert_called_once_with(max_workers=2)
        self.assertEqual(training_weights, [0, 0, 1, 1, 3, 3])
        self.assertEqual(candidate_parents, training_weights)
        self.assertEqual(best.biases[-1][0], 3)
        self.assertEqual(incumbent.biases[-1][0], 0)
        self.assertEqual((winner["score"], winner["duree"]), (48, 30))
        self.assertEqual(evaluated_weights, [0, 0, 0, 1, 2, 1, 1, 3, 4, 3, 3, 5, 6])
        self.assertEqual(recorded_winners, [3])
        lower = (60 + 0.5 * (1 - 15 / MATCH_SECONDS)) / 8
        high = (64 + 0.5 * (1 - 15 / MATCH_SECONDS)) / 8
        faster = (64 + 0.5 * (1 - 13 / MATCH_SECONDS)) / 8
        np.testing.assert_allclose(targets, [high, high, lower, faster, high, lower, faster, lower,
                                             lower, high, lower, faster, lower, lower, lower])

    def test_training_targets_favor_a_point_scored_late_over_finishing_early(self):
        incumbent = Network(np.random.default_rng(7))
        targets = []
        exploration_scores = iter(((40, 1), (41, 99)))

        def play(_simulation, *, policy, record_frames=True):
            if policy.rng is None:
                return {"score": 41, "duree": 99}
            score, duration = next(exploration_scores)
            policy.history.append((np.zeros(LAYERS[0]), 0))
            if score == 41:
                policy.history.extend((np.zeros(LAYERS[0]), 0) for _ in range(4))
                policy.history.append((np.zeros(LAYERS[0]), 8))
            return {"score": score, "duree": duration}

        def update(_network, _rows, rewards, _rng, *, learning_rate):
            targets.append(rewards)

        with (patch("src.reseau.play_match", side_effect=play), patch.object(Network, "train", update),
              patch("src.reseau.os.process_cpu_count", return_value=2, create=True),
              patch("src.reseau.ProcessPoolExecutor") as pool):
            pool.return_value.__enter__.return_value.map = map
            train_matches(2, 7, incumbent, resume=True)
        np.testing.assert_array_equal(targets[0][:5], [targets[0][0]] * 5)
        self.assertAlmostEqual(float(targets[0][0] - targets[0][-1]), 1)
        self.assertGreater(targets[0][0], targets[1][-1])

    def test_validates_initial_weights_even_when_training_records_no_decisions(self):
        incumbent = Network(np.random.default_rng(7))

        def play(_simulation, *, policy, record_frames=True):
            if policy.rng is not None:
                return {"score": 64, "duree": 12}
            return {"score": 32 if policy.network is incumbent else 40, "duree": 25}

        with patch("src.reseau.play_match", side_effect=play):
            best, winner = train_matches(1, 7, incumbent)
        self.assertIsNot(best, incumbent)
        self.assertEqual(winner["score"], 40)

    def test_training_saves_one_session_replay_even_when_existing_weights_win(self):
        incumbent = Network(np.random.default_rng(7))

        def play(_simulation, *, policy, record_frames=True):
            if policy.rng is not None:
                duration = 22.85
            else:
                duration = 19.96 if policy.network is incumbent else 24.31
            return {"score": 50, "duree": duration, "frames": [duration] if record_frames else []}

        with (patch("src.reseau.play_match", side_effect=play), patch.object(Network, "load", return_value=incumbent),
              patch.object(Network, "save") as save_model, patch("src.reseau.completion_beep"),
              patch("src.reseau.save_replay", return_value=(51, "51.json")) as save,
              patch("sys.argv", ["reseau", "--model", "reseau.npz", "train", "--matches", "1"]),
              patch("builtins.print") as output):
            from src.reseau import main

            main()
        save.assert_called_once()
        self.assertEqual(save.call_args.args[0]["frames"], [24.31])
        save_model.assert_called_once_with(Path("reseau.npz"))
        self.assertEqual([call.args[0] for call in output.call_args_list], [
            "Simulation · 1 processus pour 1 matchs d’entraînement",
            "Session · 50 points en 24.31 s · replay 51",
        ])

    def test_resume_explores_alternatives_among_the_five_best_choices(self):
        class RankedNetwork:
            def predict(self, rows):
                return np.arange(len(rows))

        simulation = Simulation()
        choices = [(1, pile, 1) for pile in range(6)]
        expected = list(reversed(choices))
        policy = MatchPolicy(RankedNetwork(), rng=np.random.default_rng(7), resume=True)
        first_choices = []
        for _ in range(100):
            ordered = policy.order(simulation, {}, choices, pickup=True)
            self.assertCountEqual(ordered, choices)
            first_choices.append(ordered[0])
        self.assertEqual(set(first_choices), set(expected[:5]))
        self.assertIn(first_choices.count(expected[0]), range(60, 81))

    def test_resume_explores_the_fastest_other_pile_while_loaded(self):
        class RankedNetwork:
            def predict(self, rows):
                return np.arange(len(rows))

        simulation = Simulation()
        choices = [(1 + pile / 10, pile, 1) for pile in range(8)]
        policy = MatchPolicy(RankedNetwork(), rng=np.random.default_rng(7), resume=True)
        first_choices = [policy.order(simulation, {2: 9}, choices, pickup=True)[0] for _ in range(100)]
        self.assertEqual(set(first_choices), {choices[index] for index in (0, 4, 5, 6, 7)})

    def test_replay_rank_prefers_score_then_total_duration(self):
        short = {"score": 40, "duree": 38}
        long = {"score": 40, "duree": 72}
        lower = {"score": 32, "duree": 15}
        self.assertGreater(_replay_rank(short), _replay_rank(long))
        self.assertGreater(_replay_rank(long), _replay_rank(lower))

    def test_network_can_finish_even_with_a_pickup_available(self):
        class FinishFirst:
            def predict(self, rows):
                return np.where(rows[:, 79] == -1, 1, 0)

        simulation = two_pile_simulation()
        result = play_match(simulation, policy=MatchPolicy(FinishFirst()))
        self.assertEqual(result["duree"], 0)
        self.assertEqual(result["score"], 10)
        replay = Replay(result)
        self.assertEqual(replay.duration, 0)
        self.assertEqual(replay.frame["score"], 10)

    def test_network_can_switch_to_deposits_with_one_pile_loaded(self):
        class StopAfterOne:
            def predict(self, rows):
                return np.where(rows[:, 79] == -1, np.where(rows[:, 6] > 0, 1, -1), 0)

        replay = play_match(two_pile_simulation(), policy=MatchPolicy(StopAfterOne()))
        actions = [frame["action"] for frame in replay["frames"]
                   if frame["action"].startswith(("Pile ", "Dépôt "))]
        self.assertEqual([action.split()[0] for action in actions], ["Pile", "Dépôt", "Pile", "Dépôt"])

    def test_network_can_choose_the_opposite_approach_to_a_centered_deposit(self):
        class PreferOppositeApproach:
            def predict(self, rows):
                pickup = -abs(rows[:, 90] - 105 / 380)
                deposit = -rows[:, 87] - 20 * (abs(rows[:, 86] - 0.2) + abs(rows[:, 85] - 1 / 12))
                return np.select((rows[:, 79] == 1, rows[:, 79] == 0), (pickup, deposit), default=-2)

        replay = play_match(Simulation(blocks=START_BLOCKS[:1]), policy=MatchPolicy(PreferOppositeApproach()))
        deposit = next(frame for frame in replay["frames"] if frame["action"].startswith("Dépôt "))
        self.assertEqual(sum(x for x, _ in deposit["piles"][0]) / 4, 250)
        self.assertLess(deposit["position"][1], 400)
        self.assertEqual(replay["score"], 18)

    def test_shifted_and_rotated_deposit_scores_and_can_be_replayed_from_saved_pose(self):
        offsets = ((60, 0), *CENTERED_OFFSETS[1:])
        angles = (radians(30), *ALIGNED_ANGLES[1:])
        target = Vec2d(310, 400)
        self.assertNotIn(target, {site for site, _ in ALTERNATE_DROPS})

        class PreferSavedDrop(MatchPolicy):
            def order(self, _simulation, _held, choices, *, pickup, **_options):
                if pickup:
                    return sorted(choices, key=lambda choice: choice[1:] != (0, 2))
                return sorted(choices, key=lambda choice: (choice[3].get_distance(target),
                                                          abs(remainder(choice[4] - angles[0], pi))))

        network = Network(np.random.default_rng(7))
        network.drop_offsets = offsets
        network.drop_angles = angles
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "reseau.npz"
            network.save(path)
            loaded = Network.load(path)
            self.assertEqual(loaded.drop_offsets, offsets)
            self.assertEqual(loaded.drop_angles, angles)
            np.savez(path, *network.weights, *network.biases, drop_offsets=np.asarray(offsets))
            legacy = Network.load(path)
            self.assertEqual(legacy.drop_offsets, offsets)
            self.assertEqual(legacy.drop_angles, ALIGNED_ANGLES)

        replay = play_match(Simulation(blocks=START_BLOCKS[:1]), policy=PreferSavedDrop(loaded))
        deposit = next(frame for frame in replay["frames"] if frame["action"].startswith("Dépôt "))
        pile = [Vec2d(*corner) for corner in deposit["piles"][0]]
        center = sum(pile, Vec2d(0, 0)) / 4
        self.assertLess(center.get_distance(target), 0.001)
        edge = max((b - a for a, b in zip(pile, pile[1:] + pile[:1])), key=lambda side: side.length)
        self.assertAlmostEqual(remainder(edge.angle - angles[0], pi), 0, places=5)
        self.assertEqual(deposit["score"], 8)
        self.assertEqual(replay["decalages_depot_mm"], offsets)
        self.assertEqual(replay["rotations_depot_rad"], angles)

    def test_features_distinguish_stock_held_piles_and_occupied_walls(self):
        sim = Simulation(blocks=(START_BLOCKS[0], START_BLOCKS[1], (90, 345, 410, 455), (1050, 945, 1370, 1055)))
        row = features(sim, {1: 1}, [None], pickup=True)[0]
        np.testing.assert_array_equal(row[11:39:7], [1, 0, 0, 0])
        np.testing.assert_array_equal(row[12:40:7], [0, 1, 0, 0])
        np.testing.assert_array_equal(row[13:41:7], [0, 0, 1, 0])
        self.assertAlmostEqual(row[78], 1 / 5)
        np.testing.assert_array_equal(row[80:], 0)

    def test_action_distance_uses_the_pickup_path_and_the_drop_destination(self):
        sim = reference_simulation(((800, 700, 950, 1300),), blocks=((1445, 840, 1555, 1160),))
        state = _action_state(sim)
        plan = _pickup_approach(sim, sim.blocks[0], 1, state)
        path = (state[0], *plan.approach.before, *plan.approach.after, plan.end)
        expected = sum(dist(a, b) for a, b in zip(path, path[1:]))
        self.assertGreater(expected, dist(path[0], path[-1]) + 100)
        row = features(sim, {}, [(plan.duration, 0, 1)], pickup=True, pickup_plans={(0, 1): plan})[0]
        self.assertAlmostEqual(row[ACTION_DISTANCE], expected / TABLE_DIAGONAL)
        self.assertTrue(take_pile(sim, pile=0, cup=1))
        site = Vec2d(250, 400)
        state = _action_state(sim)
        plan = _drop_approach(sim.blocks[0], site, 0, state)
        expected = dist(state[0], (plan.after or plan.before)[-1])
        rows = features(sim, {1: 0}, [(8, 1, 1, site, 0), (0, 1, 1, Vec2d(1500, 1000), 0)], pickup=False)
        self.assertAlmostEqual(rows[0, ACTION_DISTANCE], expected / TABLE_DIAGONAL)
        self.assertEqual(rows[1, ACTION_DISTANCE], -1)

    def test_loading_96_input_weights_preserves_predictions_on_the_official_table(self):
        rng = np.random.default_rng(7)
        network = Network(rng)
        old_weights = np.delete(network.weights[0], ACTION_DISTANCE, axis=0)
        current = rng.random((3, LAYERS[0]), dtype=np.float32)
        old = np.delete(current, ACTION_DISTANCE, axis=1)
        old[:, 11:77:7] = 1
        old[:, 78] /= 2
        hidden = np.maximum(old @ old_weights + network.biases[0], 0)
        hidden = np.maximum(hidden @ network.weights[1] + network.biases[1], 0)
        expected = (hidden @ network.weights[2] + network.biases[2])[:, 0]
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "reseau.npz"
            np.savez(path, old_weights, *network.weights[1:], *network.biases)
            migrated = Network.load(path)
        np.testing.assert_allclose(migrated.predict(current), expected, atol=1e-6)
        np.testing.assert_array_equal(migrated.weights[0][ACTION_DISTANCE], 0)


class SimulationTest(unittest.TestCase):
    def test_translation_holonome_and_time(self):
        sim = reference_simulation()
        sim.move_to(250, 2000)
        sim.advance(0.5)
        self.assertAlmostEqual(sim.position[1], 1125, delta=1)
        sim.advance(0.5)
        self.assertAlmostEqual(sim.position[1], 1500, delta=10)
        sim.advance(0.5)
        self.assertTrue(sim.moving)
        self.assertLess(sim.position[1], 2000 - SIDE / 2)
        sim.advance(0.5)
        self.assertAlmostEqual(sim.position[1], 2000 - SIDE / 2, places=5)
        self.assertFalse(sim.moving)
        for i, vertex in enumerate(FOOTPRINT):
            self.assertAlmostEqual(dist(FOOTPRINT[i - 1], vertex), SIDE)

    def test_turn_by_preserves_the_destination_while_rolling(self):
        sim = reference_simulation(blocks=())
        sim.move_to(1250, 1000)
        sim.turn_by(radians(90))
        sim.advance(0.125)
        self.assertAlmostEqual(sim.position[0], 250 + 1000 * 0.125**2 / 2, delta=0.2)
        self.assertAlmostEqual(sim.heading, radians(45))
        self.assertTrue(sim.moving)
        sim.advance(1.875)
        self.assertEqual(sim.position, (1250, 1000))
        self.assertAlmostEqual(sim.heading, radians(90))
        self.assertFalse(sim.moving)

    def test_rolling_turn_updates_the_clearance_to_walls(self):
        for wall in ((), ((0, 0, 3000, 100),)):
            with self.subTest(wall=wall):
                sim = reference_simulation(wall, blocks=())
                sim.move_to(250, 0)
                sim.turn_by(radians(90))
                sim.advance(2)
                self.assertTrue(sim.movement.blocked)
                self.assertFalse(sim.moving)
                self.assertAlmostEqual(sim.heading, radians(90))
                edge = 100 if wall else 0
                self.assertAlmostEqual(min(sim.position[1] + y for _, y in sim.footprint), edge, places=5)

    def test_rolling_turn_can_be_blocked_after_reaching_the_destination(self):
        sim = reference_simulation(blocks=())
        sim.move_to(250, 190)
        sim.advance(2)

        sim.move_to(850, 190)
        sim.turn_by(radians(-15))
        sim.advance(2)

        self.assertEqual(sim.position, (850, 190))
        self.assertEqual(sim.heading, 0)
        self.assertFalse(sim.moving)
        self.assertFalse(sim.movement.blocked)
        self.assertTrue(sim.movement.rotation_blocked)

        sim.move_to(950, 190)
        sim.advance(1)
        self.assertFalse(sim.movement.rotation_blocked)

    def test_entire_triangle_stays_inside_table_and_can_leave_the_border(self):
        sim = reference_simulation(blocks=())
        sim.move_to(1500, 0)
        sim.advance(10)
        self.assertTrue(sim.movement.blocked)
        self.assertAlmostEqual(sim.position[1], SIDE / 2, places=5)
        self.assertLess(sim.position[0], 1500)
        sim.move_to(1500, 1000)
        sim.advance(10)
        self.assertEqual(sim.position, (1500, 1000))

    def test_crossing_an_obstacle_stops_even_with_a_single_large_time_step(self):
        sim = reference_simulation(((800, 1150, 850, 1200),))
        sim.move_to(1300, 1000)
        sim.advance(10)
        # Le centre passe à côté du rectangle, mais le coin avant le touche.
        contact_x = 800 - max(x for x, _ in FOOTPRINT)
        self.assertTrue(sim.movement.blocked)
        self.assertAlmostEqual(sim.position[0], contact_x, places=5)
        sim.move_to(*START)
        sim.advance(10)
        self.assertEqual(sim.position, START)

    def test_an_obstacle_can_fit_inside_the_bounding_box_beside_the_triangle(self):
        sim = reference_simulation(((1000, 1170, 1020, 1190),))
        sim.move_to(1110, 1000)
        sim.advance(10)
        self.assertTrue(sim.movement.blocked)
        # Le robot ne peut atteindre cette pose en traversant le rectangle de front.
        sim.reset()
        sim.move_to(250, 700)
        sim.advance(10)
        sim.move_to(1110, 700)
        sim.advance(10)
        sim.move_to(1110, 1000)
        sim.advance(10)
        self.assertEqual(sim.position, (1110, 1000))

    def test_replacing_a_destination_uses_the_current_position(self):
        sim = reference_simulation()
        sim.move_to(1250, 1000)
        sim.advance(0.5)
        sim.move_to(750, 500)
        sim.advance(2)
        self.assertEqual(sim.position, (750, 500))
        sim.reset()
        self.assertEqual(sim.position, START)
        self.assertEqual(sim.time, 0)

    def test_turn_preserves_center_and_updates_collision_footprint(self):
        sim = reference_simulation()
        sim.turn_by(6 * TURN_STEP)
        self.assertEqual(sim.position, START)
        self.assertAlmostEqual(sim.heading, radians(90))
        sim.move_to(0, 1000)
        sim.advance(1)
        self.assertAlmostEqual(sim.position[0], SIDE / 2, places=5)
        self.assertTrue(sim.movement.blocked)
        sim.reset()
        self.assertEqual(sim.heading, 0)

    def test_turn_cannot_sweep_across_border_even_with_clear_endpoints(self):
        sim = reference_simulation()
        sim.turn_by(radians(-37.5))
        sim.move_to(250, 219)
        sim.advance(2)
        self.assertFalse(sim.movement.blocked)
        sim.turn_by(TURN_STEP)
        self.assertTrue(sim.movement.blocked)
        self.assertAlmostEqual(sim.heading, radians(-37.5))

    def test_turn_cannot_sweep_through_an_obstacle_with_clear_endpoints(self):
        sim = reference_simulation(((31.5, 970.5, 33.5, 972.5),))
        sim.turn_by(TURN_STEP)
        self.assertTrue(sim.movement.blocked)
        self.assertEqual(sim.heading, 0)
        sim.turn_by(-TURN_STEP)
        self.assertFalse(sim.movement.blocked)
        self.assertAlmostEqual(sim.heading, -TURN_STEP)

    def test_turn_pushes_a_free_pile_aside_in_either_direction(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                y = 1000 + side * 90
                sim = reference_simulation(blocks=((375, y - 160, 485, y + 160),))
                initial, = sim.blocks
                # L'angle positif balaie d'abord le coin haut du triangle vers la droite.
                sim.turn_by(-side * TURN_STEP)
                sim.advance(1)
                pile, = sim.blocks
                self.assertFalse(sim.moving)
                self.assertFalse(sim.movement.blocked)
                self.assertAlmostEqual(sim.heading, -side * TURN_STEP)
                self.assertLess(dist(sim.position, START), 1)
                self.assertGreater(pile.position[0] - initial.position[0], 10)

    def test_turn_stops_when_the_pile_is_pinned_by_a_wall(self):
        sim = reference_simulation(((485, 600, 550, 1150),), blocks=((375, 750, 485, 1070),))
        sim.turn_by(TURN_STEP)
        sim.advance(1)
        self.assertFalse(sim.moving)
        self.assertTrue(sim.movement.blocked)
        self.assertLess(sim.heading, TURN_STEP)
        self.assertLess(dist(sim.position, START), 1)
        self.assertLessEqual(max(x for x, _ in sim.blocks[0].corners), 486)

    def test_a_centered_push_moves_the_block_without_spinning_it(self):
        sim = reference_simulation(blocks=((800, 945, 1120, 1055),))
        initial, = sim.blocks
        sim.move_to(1250, 1000)
        sim.advance(0.25)
        self.assertEqual(sim.blocks, (initial,))
        sim.advance(1)
        pushed, = sim.blocks
        self.assertGreater(pushed.position[0], initial.position[0] + 100)
        self.assertAlmostEqual(pushed.angle, 0, delta=radians(1))

        sim.move_to(*START)
        sim.advance(2)
        self.assertEqual(sim.position, START)
        settled, = sim.blocks
        self.assertGreater(settled.position[0], pushed.position[0])
        sim.advance(1)
        self.assertEqual(sim.blocks, (settled,))

    def test_hitting_opposite_corners_spins_the_piles_in_opposite_directions(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                y = 1000 + side * 225
                sim = reference_simulation(blocks=((800, y - 55, 1120, y + 55),))
                sim.move_to(1250, 1000)
                sim.advance(2)
                block, = sim.blocks
                self.assertGreater(side * block.angle, radians(10))
                self.assertGreater(block.position[0], 960)
                for index, corner in enumerate(block.corners):
                    self.assertAlmostEqual(dist(block.corners[index - 2], corner), dist((0, 0), (320, 110)))

    def test_a_chain_of_blocks_stops_against_a_fixed_wall(self):
        wall = (1200, 900, 1300, 1100)
        sim = reference_simulation((wall,), blocks=((800, 945, 910, 1055), (1000, 945, 1110, 1055)))
        sim.move_to(1500, 1000)
        sim.advance(10)
        self.assertTrue(sim.movement.blocked)
        first, second = sim.blocks
        self.assertAlmostEqual(max(x for x, _ in second.corners), wall[0], delta=2)
        self.assertGreater(first.position[0], 1000)
        self.assertLess(sim.position[0], 885)
        self.assertFalse(sim.moving)

    def test_a_block_against_the_table_border_stops_the_robot(self):
        sim = reference_simulation()
        initial = sim.blocks[0]
        sim.move_to(750, 1000)
        sim.advance(2)
        sim.move_to(750, 0)
        sim.advance(10)
        self.assertTrue(sim.movement.blocked)
        self.assertAlmostEqual(sim.position[1], 110 + SIDE / 2, delta=2)
        block = sim.blocks[0]
        self.assertGreaterEqual(min(y for _, y in block.corners), -1)
        self.assertLess(dist(block.position, initial.position), 15)

    def test_reset_restores_the_official_blocks_during_a_push(self):
        sim = reference_simulation()
        initial = sim.blocks
        self.assertEqual(len(initial), 10)
        self.assertAlmostEqual(initial[8].angle, radians(90))
        sim.move_to(1500, 1000)
        sim.advance(2.3)
        sim.move_to(1500, 0)
        sim.advance(0.85)
        self.assertTrue(sim.moving)
        self.assertNotEqual(sim.blocks, initial)

        sim.reset()
        self.assertEqual(sim.time, 0)
        sim.advance(1)
        self.assertEqual(sim.position, START)
        self.assertEqual(sim.heading, 0)
        self.assertEqual(sim.blocks, initial)
        self.assertFalse(sim.moving)

    def test_each_cup_can_grab_off_center_and_drop_the_pile(self):
        for number, heading in ((1, 0), (2, -120), (3, 120)):
            with self.subTest(cup=number):
                sim = reference_simulation(blocks=((650, 900, 760, 1220),))
                sim.turn_by(radians(heading))
                sim.toggle_suction(number)
                sim.move_to(900, 1000)
                sim.advance(2)
                self.assertEqual([cup.enabled for cup in sim.suction_cups], [n == number for n in (1, 2, 3)])
                pile, = sim.blocks
                x, y = sim.position
                px, py = pile.position
                self.assertAlmostEqual(py - y, 60)

                sim.turn_by(radians(90))
                carried, = sim.blocks
                self.assertAlmostEqual(dist(carried.position, (x - (py - y), y + (px - x))), 0)
                self.assertAlmostEqual((carried.angle - pile.angle) % radians(360), radians(90))
                sim.toggle_suction(number)
                sim.move_to(x - 300, y)
                sim.advance(1)
                self.assertFalse(sim.suction_cups[number - 1].enabled)
                self.assertAlmostEqual(dist(sim.blocks[0].position, carried.position), 0)

    def test_drops_the_second_pile_in_zone_and_leaves_with_the_first(self):
        sim = two_pile_simulation()
        self.assertTrue(take_pile(sim, pile=0, cup=1))
        self.assertEqual(sim.held_pile(1), 0)
        self.assertFalse(take_pile(sim, pile=1, cup=1))
        self.assertTrue(take_pile(sim, pile=1, cup=2))
        self.assertEqual((sim.held_pile(1), sim.held_pile(2)), (0, 1))
        self.assertAlmostEqual(abs(sim.blocks[0].position[1] - sim.blocks[1].position[1]), 330)

        before = sim.position, sim.heading, sim.time, sim.blocks
        self.assertFalse(place_pile(sim, cup=2, x=1600, y=1200, angle=radians(90)))
        self.assertEqual((sim.position, sim.heading, sim.time, sim.blocks), before)
        for x, y in TOWER_CENTERS:
            self.assertFalse(place_pile(sim, cup=2, x=x, y=y, angle=0))
            self.assertEqual((sim.position, sim.heading, sim.time, sim.blocks), before)
        release = []
        self.assertTrue(place_pile(sim, cup=2, x=850, y=1000, angle=radians(90),
                                   on_release=lambda: release.append(sim.position)))
        self.assertAlmostEqual(dist(sim.position, release[0]), 50)
        self.assertAlmostEqual(dist(sim.blocks[1].position, (850, 1000)), 0)
        self.assertAlmostEqual(sim.blocks[1].angle, radians(90))
        self.assertEqual((sim.held_pile(1), sim.held_pile(2)), (0, None))
        self.assertEqual(sim.score, 8)

        first_offset = Vec2d(*sim.blocks[0].position) - Vec2d(*sim.position)
        deposited = sim.blocks[1]
        exit_position = (sim.position[0] - 250, sim.position[1])
        sim.move_to(*exit_position)
        sim.advance(1.1)
        self.assertEqual(sim.position, exit_position)
        self.assertEqual((sim.held_pile(1), sim.held_pile(2)), (0, None))
        self.assertAlmostEqual(dist(sim.blocks[0].position, Vec2d(*exit_position) + first_offset), 0)
        self.assertEqual(sim.blocks[1], deposited)
        self.assertEqual(sim.score, 8)

    def test_can_back_away_from_the_table_border_with_a_pile_and_drop_it(self):
        sim = reference_simulation()
        self.assertTrue(take_pile(sim, pile=4, cup=2))
        pile = sim.blocks[4]
        cup_x, _ = sim.suction_cups[1].position
        self.assertAlmostEqual(cup_x, pile.position[0])
        self.assertAlmostEqual(max(y for _, y in pile.corners), 2000)

        initial_y = sim.position[1]
        self.assertTrue(place_pile(sim, cup=2, x=675, y=1425, angle=-pi / 4))
        self.assertLess(sim.position[1], initial_y)
        self.assertIsNone(sim.held_pile(2))
        self.assertAlmostEqual(dist(sim.blocks[4].position, (675, 1425)), 0)
        self.assertEqual(sim.score, 8)

    def test_pickup_starts_at_25_mm_and_advances_up_to_50_mm(self):
        wall = ((1565, 750, 1600, 1250),)
        for right, obstacles, shift in ((1555, (), 27), (3000, (), 0), (2990, (), 10), (1555, wall, 10)):
            with self.subTest(right_edge=right, obstacles=obstacles):
                sim = reference_simulation(obstacles, blocks=((right - 110, 840, right, 1160),))
                initial = sim.blocks[0].position
                trace = []

                def advance(seconds):
                    sim.advance(seconds)
                    trace.append((sim.suction_cups[0].position, sim.held_pile(1)))

                self.assertTrue(take_pile(sim, pile=0, cup=1, advance=advance))
                standoff_x = right - 110 - 25
                self.assertTrue(any(abs(point[0] - standoff_x) < 1e-6 and held is None for point, held in trace))
                self.assertEqual(sim.held_pile(1), 0)
                self.assertAlmostEqual(sim.suction_cups[0].position[0] - standoff_x, 23 + shift, places=5)
                self.assertAlmostEqual(sim.blocks[0].position[0] - initial[0], shift, places=5)
                boundary = obstacles[0][0] if obstacles else 3000
                self.assertLessEqual(max(x for x, _ in sim.blocks[0].corners), boundary + 1e-6)

    def test_deposit_can_move_diagonally_away_before_taking_the_left_corridor(self):
        robot = RobotGeometry(FOOTPRINT, (
            ((FRONT, -100), Vec2d(1, 0)),
            ((-FRONT / 2, 95), Vec2d(1, 0).rotated(radians(120))),
            ((-FRONT / 2, -95), Vec2d(1, 0).rotated(radians(-120))),
            ((FRONT, 100), Vec2d(1, 0)),
        ))
        sim = Simulation(robot=robot)
        self.assertTrue(take_pile(sim, pile=8, cup=1))
        self.assertTrue(take_pile(sim, pile=9, cup=2))
        self.assertTrue(place_pile(sim, cup=2, x=675, y=575, angle=pi / 4))
        self.assertTrue(place_pile(sim, cup=1, x=850, y=1000, angle=pi / 2))
        self.assertTrue(take_pile(sim, pile=4, cup=2))
        self.assertTrue(take_pile(sim, pile=1, cup=4))
        self.assertTrue(place_pile(sim, cup=4, x=675, y=1425, angle=-pi / 4))

        self.assertTrue(place_pile(sim, cup=2, x=250, y=400, angle=0))
        self.assertAlmostEqual(dist(sim.blocks[4].position, (250, 400)), 0)
        self.assertEqual(sim.score, 32)

    def test_separate_actions_check_the_route_before_touching_the_load(self):
        wall = (900, 0, 950, 2000)
        sim = reference_simulation((wall,), blocks=((650, 900, 760, 1220), (1445, 340, 1555, 660)))
        self.assertTrue(take_pile(sim, pile=0, cup=1))
        before = sim.position, sim.heading, sim.time, sim.blocks
        self.assertFalse(take_pile(sim, pile=1, cup=2))
        self.assertEqual((sim.position, sim.heading, sim.time, sim.blocks), before)
        self.assertFalse(place_pile(sim, cup=1, x=1040, y=1000, angle=radians(90)))
        self.assertEqual((sim.position, sim.heading, sim.time, sim.blocks), before)
        self.assertEqual(sim.held_pile(1), 0)

    def test_separate_deposit_turns_the_robot_to_place_the_pile_at_the_requested_angle(self):
        sim = reference_simulation(blocks=((1445, 340, 1555, 660),))
        self.assertTrue(take_pile(sim, pile=0, cup=1))
        self.assertTrue(place_pile(sim, cup=1, x=850, y=1000, angle=0))
        self.assertIsNone(sim.held_pile(1))
        self.assertAlmostEqual(dist(sim.blocks[0].position, (850, 1000)), 0)
        self.assertAlmostEqual(sim.blocks[0].angle, 0)
        self.assertEqual(sim.score, 8)

    def test_cup_does_not_grab_the_short_side(self):
        sim = reference_simulation(blocks=((650, 945, 970, 1055),))
        initial = sim.blocks
        sim.move_to(650 - FRONT - 2, START[1])
        sim.advance(1)
        sim.toggle_suction(1)
        sim.move_to(*START)
        sim.advance(1)
        self.assertEqual(sim.blocks, initial)

    def test_cup_accepts_small_misalignment_at_contact(self):
        for degrees, holding in ((10, True), (30, False)):
            with self.subTest(degrees=degrees):
                angle = radians(degrees)
                front_x = FOOTPRINT[1][0]
                top = 1000 + front_x * sin(angle) - 2
                sim = reference_simulation(blocks=((650, top, 760, top + 320),))
                sim.turn_by(angle)
                sim.move_to(650 - front_x * cos(angle) - 2, 1000)
                sim.advance(1)
                # On active à l'arrêt pour que la poussée ne réaligne pas la pile.
                sim.toggle_suction(1)
                sim.move_to(*START)
                sim.advance(1)
                self.assertEqual(sim.blocks[0].position[0] < 600, holding)

    def test_held_pile_blocks_movement_and_reset_releases_it(self):
        wall = (1200, 850, 1250, 1250)
        sim = reference_simulation((wall,), blocks=((650, 900, 760, 1220),))
        initial = sim.blocks
        sim.toggle_suction(1)
        sim.move_to(1450, 1000)
        sim.advance(2)
        self.assertTrue(sim.movement.blocked)
        self.assertAlmostEqual(max(x for x, _ in sim.blocks[0].corners), wall[0], places=5)
        self.assertLess(sim.position[0] + FOOTPRINT[1][0], wall[0] - 100)
        sim.turn_by(TURN_STEP)
        self.assertTrue(sim.movement.blocked)
        self.assertEqual(sim.heading, 0)

        sim.reset()
        self.assertEqual(sim.blocks, initial)
        self.assertFalse(any(cup.enabled for cup in sim.suction_cups))

    def test_score_follows_a_pile_pushed_into_and_out_of_a_wall_zone(self):
        sim = reference_simulation(blocks=((550, 870, 660, 1130),))
        self.assertEqual(sim.score, 0)

        sim.move_to(715, 1000)
        sim.advance(1)
        self.assertEqual(sim.score, 8)

        sim.move_to(1020, 1000)
        sim.advance(1)
        self.assertEqual(sim.score, 0)

    def test_score_follows_a_released_pile_at_the_edge_of_a_wall_zone(self):
        sim = reference_simulation(blocks=((630, 840, 740, 1160),))
        self.assertEqual(sim.score, 0)
        sim.toggle_suction(1)
        sim.move_to(900, 1000)
        sim.advance(1)
        x, y = sim.position
        px, py = sim.blocks[0].position
        sim.move_to(695 + x - px, 1000 + y - py)
        sim.advance(1)
        self.assertEqual(sim.score, 0)
        sim.toggle_suction(1)
        self.assertEqual(sim.score, 8)

        sim.toggle_suction(1)
        self.assertEqual(sim.score, 0)
        x, y = sim.position
        sim.move_to(x - 1, y)
        sim.advance(0.1)
        sim.toggle_suction(1)
        self.assertEqual(sim.score, 0)

    def test_two_free_piles_in_one_zone_only_get_one_wall_bonus(self):
        sim = reference_simulation(blocks=((735, 840, 845, 1160), (855, 840, 965, 1160)))
        self.assertEqual(sim.score, 11)

    def test_score_counts_stones_in_a_tower_circle_without_a_wall_bonus(self):
        sim = reference_simulation(blocks=((650, 900, 760, 1220),))
        sim.toggle_suction(1)
        sim.move_to(900, 1000)
        sim.advance(2)
        sim.turn_by(radians(-80))
        x, y = sim.position
        px, py = sim.blocks[0].position
        sim.move_to(480 + x - px, 240 + y - py)
        sim.advance(2.5)
        sim.toggle_suction(1)
        self.assertEqual(sim.score, 3)


class AutomaticMatchTest(unittest.TestCase):
    def test_planned_rolling_turn_uses_the_clearance_in_the_requested_direction(self):
        for edge in (0, 100):
            with self.subTest(edge=edge):
                wall = ((0, 0, 3000, edge),) if edge else ()
                sim = reference_simulation(wall, blocks=())
                sim.move_to(250, edge + 200)
                sim.advance(2)
                start, footprints, obstacles = _action_state(sim)
                end = Vec2d(850, edge + 200)
                angle = radians(15)

                plan = _approach(start, end, angle, footprints, obstacles)
                self.assertIsNotNone(plan)
                self.assertTrue(plan.rolling)
                self.assertIsNone(_approach(start, end, -angle, footprints, obstacles))
                self.assertTrue(_follow(sim, plan))
                self.assertEqual(sim.position, tuple(end))
                self.assertAlmostEqual(sim.heading, angle)

    def test_planned_rolling_turn_checks_intermediate_angles(self):
        sim = reference_simulation(blocks=())
        sim.turn_by(radians(75))
        sim.move_to(250, 215)
        sim.advance(2)
        start, footprints, obstacles = _action_state(sim)
        end = Vec2d(850, 215)

        plan = _approach(start, end, radians(30), footprints, obstacles)
        self.assertIsNotNone(plan)
        self.assertFalse(plan.rolling)
        self.assertTrue(_follow(sim, plan))
        self.assertEqual(sim.position, tuple(end))
        self.assertAlmostEqual(sim.heading, radians(105))

    def test_route_uses_multiple_passages_and_skips_unnecessary_waypoints(self):
        start, end = Vec2d(300, 1000), Vec2d(2700, 1000)
        footprints = (rectangle_corners((-20, -20, 20, 20)),)
        rectangles = ((800, 0, 950, 1350), (1650, 650, 1800, 2000))
        obstacles = tuple((rectangle_corners(bounds), bounds) for bounds in rectangles)
        path, duration = _route(start, end, footprints, obstacles)
        nodes = (start, *path)
        self.assertEqual(nodes[-1], end)
        self.assertEqual(len(path), 3)
        self.assertTrue(all(_clear(a, b, footprints, obstacles) for a, b in zip(nodes, nodes[1:])))
        self.assertTrue(all(not _clear(a, b, footprints, obstacles) for a, b in zip(nodes, nodes[2:])))
        self.assertAlmostEqual(duration, sum(travel_time(a.get_distance(b)) for a, b in zip(nodes, nodes[1:])))
        self.assertEqual(_route(start, end, footprints, obstacles, duration - 0.01), ((), 0))
        self.assertEqual(_route(start, end, footprints, ())[0], (end,))

    def test_route_uses_nearby_corners_and_keeps_clearance_from_a_pile(self):
        start, end = Vec2d(1200, 200), Vec2d(2000, 200)
        footprints = (rectangle_corners((-30, -20, 30, 20)),)
        pile = (1450, 140, 1770, 225)
        obstacles = ((rectangle_corners(pile), pile),)
        path, duration = _route(start, end, footprints, obstacles)
        self.assertEqual(path[-1], end)
        self.assertEqual(len(path), 2)
        self.assertLess(sum(dist(a, b) for a, b in zip((start, *path), path)), 850)
        expanded = (pile[0] - ROUTE_MARGIN, pile[1] - ROUTE_MARGIN,
                    pile[2] + ROUTE_MARGIN, pile[3] + ROUTE_MARGIN)
        self.assertTrue(all(_clear(a, b, footprints, ((rectangle_corners(expanded), expanded),))
                            for a, b in zip((start, *path), path)))
        self.assertLess(duration, 2.6)

    def test_official_table_deposits_both_piles_carried_on_the_front_face(self):
        simulation = Simulation()
        self.assertTrue(take_pile(simulation, pile=8, cup=1))
        self.assertTrue(take_pile(simulation, pile=9, cup=4))
        self.assertTrue(place_pile(simulation, cup=1, x=250, y=400, angle=0))
        self.assertEqual(simulation.held_pile(4), 9)
        self.assertTrue(place_pile(simulation, cup=4, x=850, y=1000, angle=pi / 2))
        self.assertEqual(simulation.score, 16)

    def test_strategy_can_fill_all_four_cups(self):
        blocks = tuple(START_BLOCKS[i] for i in (0, 4, 6, 8))
        replay = play_match(Simulation(blocks=blocks), batches=True)
        self.assertEqual(max(sum(enabled for _, enabled in frame["cups"]) for frame in replay["frames"]), 4)

    def test_strategy_can_score_another_pile_in_an_occupied_wall_and_return(self):
        class PreferSameWall:
            def order(self, _simulation, _held, choices, *, pickup, **_options):
                if pickup:
                    return choices
                # Demande d'abord le centre occupé, puis le dépôt le plus proche sur ce côté.
                return sorted(choices, key=lambda choice: (choice[3].get_distance((250, 445)), choice[1]))

            def observe(self, _choice):
                pass

        simulation = Simulation(blocks=(START_BLOCKS[0], (90, 345, 410, 455)))
        self.assertEqual(simulation.score, 8)
        replay = play_match(simulation, policy=PreferSameWall())
        placed, existing = simulation.blocks
        self.assertEqual(scoring_zones(placed), ((0,), True))
        self.assertEqual(scoring_zones(existing), ((0,), True))
        self.assertLess(dist(placed.position, (250, 520)), 0.001)
        self.assertEqual(existing.position, (250, 400))
        self.assertEqual(simulation.score, 11)
        self.assertEqual(replay["score_retour"], 10)
        self.assertLess(replay["duree"], MATCH_SECONDS)

    def test_deposit_angles_cover_half_a_turn_at_each_position(self):
        self.assertEqual(len(ALTERNATE_DROPS), 4 * 18 * len(CENTERED_DROPS))
        poses = ALTERNATE_DROPS[:len(ALTERNATE_DROPS) // len(CENTERED_DROPS)]
        self.assertEqual(len({site for site, _ in poses}), 4)
        for site in {position for position, _ in poses}:
            self.assertEqual([angle for position, angle in poses if position == site],
                             [radians(angle) for angle in range(-90, 90, 10)])

    def test_match_rounds_angles_from_an_older_network_to_the_current_grid(self):
        network = Network(np.random.default_rng(0))
        network.drop_angles = (radians(16), *ALIGNED_ANGLES[1:])
        replay = play_match(Simulation(blocks=()), policy=MatchPolicy(network))
        self.assertEqual(replay["rotations_depot_rad"], (radians(20), *ALIGNED_ANGLES[1:]))
        self.assertEqual(network.drop_angles[0], radians(16))

    def test_strategy_can_turn_a_pile_50_degrees_at_200_mm_from_the_wall_center(self):
        class PreferDiagonalDrop:
            def order(self, _simulation, _held, choices, *, pickup, **_options):
                if pickup:
                    return choices
                return sorted(choices, key=lambda choice: (choice[3].get_distance((1010, 880)),
                                                          abs(remainder(choice[4] - radians(140), pi))))

            def observe(self, _choice):
                pass

        simulation = Simulation(blocks=(START_BLOCKS[8], (705, 840, 815, 1160)))
        self.assertEqual(simulation.score, 8)
        replay = play_match(simulation, policy=PreferDiagonalDrop())
        placed, existing = simulation.blocks
        self.assertLess(dist(placed.position, (1010, 880)), 0.001)
        self.assertAlmostEqual(remainder(placed.angle - radians(140), pi), 0)
        self.assertEqual(scoring_zones(placed), ((2,), True))
        self.assertEqual(existing.position, (760, 1000))
        self.assertEqual(simulation.score, 11)
        self.assertEqual(replay["score_retour"], 10)

    def test_network_learns_and_reloads_its_action_values(self):
        rng = np.random.default_rng(7)
        rows = features(two_pile_simulation(), {}, [(1, 0, 1), (1, 1, 2)], pickup=True)
        self.assertEqual(rows.shape, (2, 97))
        network = Network(rng)
        targets = np.array([1.0, 0.0], dtype=np.float32)
        initial_error = np.mean((network.predict(rows) - targets) ** 2)
        network.train(rows, targets, rng)
        self.assertLess(np.mean((network.predict(rows) - targets) ** 2), initial_error)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            path = Path(folder) / "reseau.npz"
            network.save(path)
            np.testing.assert_array_equal(Network.load(path).predict(rows), network.predict(rows))

    def test_strategy_loads_two_piles_before_scoring_with_separate_deposits(self):
        simulation = two_pile_simulation()
        replay = Replay(play_match(simulation, batches=True))

        def state(frame):
            return sum(enabled for _, enabled in frame["cups"]), frame["score"]

        self.assertEqual([value for value, _ in groupby(replay.frames[:-1], state)],
                         [(0, 0), (1, 0), (2, 0), (1, 8), (0, 16)])
        deposits = [frame for frame in replay.frames if frame["action"].startswith("Dépôt ")]
        self.assertEqual(list(map(state, deposits)), [(1, 8), (0, 16)])
        self.assertEqual(len({scoring_zones(block)[0] for block in simulation.blocks}), 2)
        self.assertLess(replay.duration, MATCH_SECONDS)

    def test_interrupted_save_leaves_only_the_previous_replay(self):
        def interrupt_write(_replay, file, **_options):
            file.write('{"frames":')
            file.flush()
            self.assertEqual(list(directory.glob("essai-*.json")), [first])
            raise KeyboardInterrupt

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            directory = Path(folder)
            replay = play_match(reference_simulation(blocks=()))
            _, first = save_replay(replay, directory)
            with patch("src.match.json.dump", side_effect=interrupt_write):
                with self.assertRaises(KeyboardInterrupt):
                    save_replay(replay, directory)
            self.assertEqual(list(directory.iterdir()), [first])
            self.assertEqual(Replay(json.loads(first.read_text(encoding="utf-8"))).duration, replay["duree"])
            self.assertEqual(save_replay(replay, directory)[0], 2)

    def test_replay_starts_at_first_timestamp(self):
        simulation = reference_simulation(blocks=())
        simulation.advance(5)
        replay = Replay(play_match(simulation))
        self.assertEqual(replay.time, 5)
        self.assertEqual(replay.frame["t"], 5)
        replay.time = 0
        self.assertEqual(replay.frame["t"], 5)

    def test_return_bonus_at_the_buzzer_depends_on_the_chassis_overlap(self):
        for x, bonus in ((START[0], 10), (650, 5), (850, 0)):
            with self.subTest(x=x):
                simulation = reference_simulation(blocks=())
                simulation.move_to(x, START[1])
                simulation.advance(MATCH_SECONDS)
                result = play_match(simulation, record_frames=False)
                self.assertEqual(result["score_retour"], bonus)
                self.assertEqual(result["score"], bonus)

    def test_return_goes_around_a_blocked_entrance(self):
        simulation = reference_simulation(((490, 750, 510, 1250),), blocks=())
        for target in ((250, 550), (1050, 550), (1050, 950)):
            simulation.move_to(*target)
            simulation.advance(2)
        self.assertEqual(simulation.position, (1050, 950))
        result = play_match(simulation, batches=True)
        self.assertEqual(result["score_retour"], 10)
        self.assertTrue(any(frame["position"][1] < 750 or frame["position"][1] > 1250
                            for frame in result["frames"] if frame["action"].startswith("Retour")))

    def test_replay_rejects_a_later_frame_with_a_missing_field(self):
        result = play_match(reference_simulation(blocks=((1445, 340, 1555, 660),)))
        del result["frames"][-1]["target"]
        with self.assertRaisesRegex(ValueError, "image [0-9]+ .*target.* manquant"):
            Replay(result)

    def test_plays_one_delivery_and_reopens_numbered_attempts(self):
        from src.match import main

        simulation = reference_simulation(blocks=((1445, 340, 1555, 660),))
        result = play_match(simulation)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            directory = Path(folder)
            first = save_replay(result, directory)
            with patch("src.match._trials", return_value={}):
                second = save_replay(result, directory, descriptive=True)
                third = save_replay(result, directory)
            self.assertEqual((first[0], second[0], third[0]), (1, 2, 3))
            self.assertEqual(second[1].name, f"essai-0002-{result['score']}pts-{result['duree']:.2f}s.json")
            self.assertEqual(set(directory.iterdir()), {first[1], second[1], third[1]})
            with patch("src.match.REPLAY_DIR", directory), patch("src.interface.create_window") as window:
                for number in (1, 2):
                    with patch("sys.argv", ["match", "replay", str(number)]):
                        main()
                    self.assertEqual(window.call_args.kwargs["trial"], number)
                replay = window.call_args.kwargs["replay"]
        self.assertEqual(replay.frame["score"], 0)
        self.assertTrue(any(frame["action"].startswith("Prise") for frame in replay.frames))
        self.assertTrue(any(frame["action"].startswith("Dépôt") for frame in replay.frames))
        self.assertTrue(any(frame["action"].startswith("Retour") for frame in replay.frames[:-1]))
        self.assertEqual(replay.times, sorted(set(replay.times)))
        self.assertEqual(replay.duration, result["duree"])
        self.assertLess(replay.duration, MATCH_SECONDS)
        self.assertFalse(simulation.moving)
        replay.time = replay.duration
        self.assertEqual(replay.frames[-2]["score"], 8)
        self.assertEqual(replay.frame["score"], 18)
        self.assertEqual(result["score_retour"], 10)
        self.assertEqual(simulation.score, 8)


class RobotConfigurationTest(unittest.TestCase):
    def test_default_equilateral_robot_has_two_front_and_two_side_cups(self):
        robot = load_robot()
        a, b, c = robot.contour
        self.assertTrue(all(abs(dist(p, q) - SIDE) < 1e-6 for p, q in ((a, b), (b, c), (c, a))))
        self.assertEqual(len(robot.mounts), 4)
        for (mount, facing), y, angle in zip(robot.mounts, (-100, 105, -105, 100), (0, 120, -120, 0), strict=True):
            self.assertEqual(mount[1], y)
            self.assertAlmostEqual(facing.get_distance(Vec2d(1, 0).rotated(radians(angle))), 0)

    def test_custom_rectangle_and_cup_drive_collisions_and_grip(self):
        config = """
contour_mm = [[-120, -80], [120, -80], [120, 80], [-120, 80]]
[[ventouses]]
position_mm = [120, 40]
angle_deg = 0
"""
        with patch.object(Path, "open", mock_open(read_data=config.encode())):
            robot = load_robot(Path("rectangle.toml"))
        sim = Simulation(((1000, 800, 1100, 1200),), blocks=((600, 900, 710, 1220),), robot=robot)
        self.assertEqual(sim.footprint, robot.contour)
        self.assertEqual([cup.position for cup in sim.suction_cups], [(370, 1040)])
        with self.assertRaises(ValueError):
            sim.toggle_suction(2)
        sim.toggle_suction(1)
        sim.move_to(1200, 1000)
        sim.advance(2)
        self.assertTrue(sim.movement.blocked)
        self.assertAlmostEqual(max(x for x, _ in sim.blocks[0].corners), 1000, places=5)
        self.assertLess(sim.position[0] + 120, 900)
        sim.reset()
        self.assertEqual([cup.position for cup in sim.suction_cups], [(370, 1040)])
        self.assertFalse(sim.suction_cups[0].enabled)

    def test_concave_contour_is_rejected(self):
        config = b"contour_mm = [[-100, -100], [100, -100], [0, 0], [100, 100], [-100, 100]]"
        with patch.object(Path, "open", mock_open(read_data=config)):
            with self.assertRaisesRegex(ValueError, "convexe"):
                load_robot(Path("concave.toml"))


if __name__ == "__main__":
    unittest.main()
