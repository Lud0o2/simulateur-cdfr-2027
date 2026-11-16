from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from src.interface import Playback
from src.match import MATCH_SECONDS, Replay, main as match_main, play_match
from src.neat import (CHAMPION_FILE, MAX_HIDDEN, MAX_LINKS, N_INPUTS, Genome, Individual, Settings, fitness,
                      initial_genome, latest_directory, mutate, offspring, read_champion, remember_strategies,
                      train, write_champion)
from src.simulateur import Simulation, WALL_ZONES


def empty_match():
    return play_match(Simulation(blocks=()), full_match=True)


def match_at_65(time):
    match = empty_match()
    first, last = match["frames"]
    match.update(score=65, score_max=65, temps_score_max=time,
                 frames=[first, dict(last, t=time, score=65), dict(last, score=65)])
    return match


class FitnessTest(unittest.TestCase):
    def test_match_tracks_first_peak_even_if_points_are_lost(self):
        def build(simulation, record):
            simulation.advance(1)
            for pile, zone in zip(simulation._piles, WALL_ZONES, strict=False):
                pile.body.position = tuple(sum(axis) / 4 for axis in zip(*zone))
            record("Deux dépôts")
            simulation.advance(1)
            record("Même score")
            for index, pile in enumerate(simulation._piles):
                pile.body.position = (2200, 500 + 500 * index)
            record("Piles déplacées")

        results = []
        for rendering in (False, True):
            with patch("src.match._play_reference", side_effect=build):
                results.append(play_match(Simulation(blocks=((2000, 700, 2320, 810), (2000, 1200, 2320, 1310))),
                                          full_match=True, record_frames=rendering))
        for result in results:
            self.assertEqual(fitness(result), (16, -1))
            self.assertEqual(result["duree"], MATCH_SECONDS)
            self.assertEqual(result["score"], 10)
        self.assertEqual(max((frame["score"], -frame["t"]) for frame in results[1]["frames"]), (16, -1))

    def test_return_is_preserved_to_the_buzzer(self):
        result = empty_match()
        self.assertEqual(fitness(result), (10, 0))
        self.assertEqual(result["score_retour"], 10)
        self.assertEqual([(frame["t"], frame["score"]) for frame in result["frames"]], [(0, 10), (100, 10)])


class GenomeTest(unittest.TestCase):
    def test_mutation_changes_weights_biases_and_adds_topology_with_limits(self):
        rng = Random(17)
        settings = Settings(add_node=1, add_link=1)
        original = initial_genome(rng, 0)
        genome = mutate(original, rng, settings, 0.3)
        self.assertEqual(len(genome.biases), 2)
        self.assertNotEqual(genome.biases[-1], original.biases[-1])
        self.assertEqual(len(genome.links), len(original.links) + 2)
        for _ in range(200):
            genome = mutate(genome, rng, settings, 0.3)
        self.assertEqual(len(genome.biases), MAX_HIDDEN + 1)
        self.assertLessEqual(len(genome.links), MAX_LINKS)
        self.assertEqual(Genome.read(dict(biases=list(genome.biases), links=list(map(list, genome.links)))), genome)

    def test_loader_rejects_cycles_duplicate_links_and_nonfinite_parameters(self):
        for links in ([[N_INPUTS, N_INPUTS, 1]], [[0, 1, 1]], [[0, N_INPUTS, float("nan")]],
                      [[0, N_INPUTS, 1], [0, N_INPUTS, 2]]):
            with self.subTest(links=links), self.assertRaises(ValueError):
                Genome.read(dict(biases=[0], links=links))

    def test_mutation_can_reset_connection_weights_without_perturbation(self):
        rng = Random(17)
        settings = Settings(add_node=0, add_link=0)
        genome = Genome((0.0,), ((0, N_INPUTS, 2.0), (1, N_INPUTS, 2.0)))
        with (patch.object(rng, "gauss", return_value=0.0),
              patch.object(rng, "random", side_effect=(0.009, 0.25, 0.01, 0.5, 0.5))):
            child = mutate(genome, rng, settings, 0.3)
        self.assertEqual(child, Genome(genome.biases, ((0, N_INPUTS, -0.5), (1, N_INPUTS, 2.0))))

    def test_offspring_combines_archived_parents_tournaments_and_new_weights(self):
        settings = Settings(population=200)
        population = [Individual(i, 1, initial_genome(Random(i), 0.8), (65 if i <= 40 else 64, -i))
                      for i in range(1, 201)]
        archive = population[:3]
        rng = Random(17)
        with (patch.object(rng, "choices", wraps=rng.choices) as draw,
              patch("src.neat.initial_genome", wraps=initial_genome) as random_genomes,
              patch("src.neat.mutate", wraps=mutate) as mutations):
            children = offspring(population, rng, settings, 2, 201, archive)
        self.assertEqual(draw.call_args_list[0].args, (archive,))
        self.assertEqual(draw.call_args_list[0].kwargs, {"k": 45})
        expected_rng = Random(17)
        selected = expected_rng.choices(archive, k=45)
        selected.extend(max(expected_rng.choices(population, k=3), key=lambda item: item.rank)
                        for _ in range(135))
        self.assertEqual([call.args[0] for call in mutations.call_args_list[:180]],
                         [parent.genome for parent in selected])
        self.assertEqual(random_genomes.call_count, 20)
        self.assertEqual(mutations.call_count, 200)
        self.assertEqual(len(children), settings.population)
        self.assertTrue(all(child.rank is None for child in children))
        self.assertEqual(children, offspring(population, Random(17), settings, 2, 201, archive))

    def test_strategy_archive_keeps_fastest_per_route_and_evicts_oldest(self):
        genome = initial_genome(Random(17), 0)
        original = Individual(1, 1, genome, (65, -60), "A")
        faster = Individual(2, 1, genome, (65, -50), "A")
        slower = Individual(3, 1, genome, (65, -70), "A")
        other = Individual(4, 1, genome, (65, -55), "B")
        recent = Individual(5, 2, genome, (65, -65), "C")
        archive = {}
        with patch("src.neat.STRATEGY_ARCHIVE_LIMIT", 2):
            remember_strategies(archive, [other, faster, slower, original])
            self.assertEqual(archive, {"A": faster, "B": other})
            remember_strategies(archive, [recent])
        self.assertEqual(archive, {"B": other, "C": recent})


class TrainingTest(unittest.TestCase):
    def test_atomic_champion_save_and_load(self):
        match = empty_match()
        individual = Individual(1, 1, initial_genome(Random(3), 0), fitness(match))
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            write_champion(directory, individual, match, Settings(seed=13))
            with patch("src.neat.json.dump", side_effect=OSError("Interruption")):
                with self.assertRaises(OSError):
                    write_champion(directory, individual, match, Settings(seed=14))
            self.assertEqual([file.name for file in directory.iterdir()], [CHAMPION_FILE])
            data, genome, replay = read_champion(directory)
            self.assertEqual(data["settings"]["seed"], 13)
            self.assertEqual(genome, individual.genome)
            self.assertEqual(replay.frames, match["frames"])

    def test_default_replays_skip_saved_duels(self):
        match = empty_match()
        duel_match = dict(match, score_adverse=18)
        individual = Individual(1, 1, initial_genome(Random(3), 0), fitness(match))
        with TemporaryDirectory() as folder:
            root = Path(folder)
            solo, duel = root / "session-1", root / "session-2"
            solo.mkdir()
            duel.mkdir()
            champion = write_champion(solo, individual, match, Settings(seed=13))
            (duel / CHAMPION_FILE).write_text(json.dumps(dict(champion, scenario="duel", replay=duel_match)),
                                             encoding="utf-8")
            (root / "essai-0001.json").write_text(json.dumps(match), encoding="utf-8")
            (root / "essai-0002.json").write_text(json.dumps(duel_match), encoding="utf-8")
            with patch("src.neat.RUN_DIR", root):
                self.assertEqual(latest_directory(), solo)
            with (patch("src.match.REPLAY_DIR", root), patch("src.interface.create_window") as window,
                  patch("sys.argv", ["match", "replay"])):
                match_main()
            self.assertEqual(window.call_args.kwargs["trial"], 1)
            with self.assertRaisesRegex(ValueError, "incompatible avec le mode solo"):
                Replay(duel_match)

    def test_only_qualified_genomes_can_become_champion(self):
        events = []
        pause = Event()
        pause.set()
        match = match_at_65(30)
        ranks = iter(((64, -1), (60, -2), (65, -30), (64, -3)))

        def publish(event):
            events.append(event)
            if event["status"] == "Pause":
                self.assertEqual(event["simulations"], 0)
                pause.clear()

        def evaluate(_genome, record_frames=False):
            if record_frames:
                return match
            rank = next(ranks)
            return rank, "route" if rank[0] == 65 else None

        with (TemporaryDirectory() as folder, patch("src.neat.ProcessPoolExecutor", ThreadPoolExecutor),
              patch("src.neat.evaluate", side_effect=evaluate), patch("src.neat.offspring", wraps=offspring) as reproduce):
            directory = Path(folder)
            replays = directory / "replays"
            with patch("src.neat.REPLAY_DIR", replays):
                train(Settings(population=2, generations=2, seed=13), directory, publish, pause=pause)
            history = [json.loads(line) for line in (directory / "generations.jsonl").read_text().splitlines()]
            data, _, _ = read_champion(directory)
            self.assertEqual([path.name for path in replays.iterdir()], ["essai-0001-65pts-30.00s.json"])
            with (patch("src.match.REPLAY_DIR", replays), patch("src.interface.create_window") as window,
                  patch("sys.argv", ["match", "replay", "1"])):
                match_main()
            self.assertEqual(window.call_args.kwargs["trial"], 1)
            self.assertEqual(window.call_args.kwargs["replay"].frames, match["frames"])
        self.assertEqual([item.id for item in reproduce.call_args.args[0]], [1, 2])
        self.assertEqual([(stats["simulations"], stats["qualified"]) for stats in history], [(2, 0), (4, 1)])
        self.assertEqual(data["generation"], 2)
        self.assertEqual(data["id"], 3)
        self.assertEqual(events[-1]["status"], "Terminé")
        self.assertIn("Pause", [event["status"] for event in events])

    def test_next_generation_starts_during_replay_and_stop_saves_latest_champion(self):
        stop = Event()
        times = {}
        first_recording = None

        def submit(_evaluate, genome, record_frames=False):
            nonlocal first_recording
            future = Future()
            if record_frames:
                replay = match_at_65(times[genome])
                if first_recording is None:
                    first_recording = future, replay
                else:
                    future.set_result(replay)
            else:
                time = times[genome] = 50 - len(times)
                if len(times) == 4:
                    pending, replay = first_recording
                    self.assertFalse(pending.done())
                    pending.set_result(replay)
                future.set_result(((65, -time), str(time)))
            return future

        def wait(futures, **_options):
            ready = {future for future in futures if future.done()}
            self.assertTrue(ready, "La génération suivante attend le replay")
            return ready, set(futures) - ready

        def publish(event):
            if event["simulations"] == 4:
                stop.set()

        with (TemporaryDirectory() as folder, patch("src.neat.ProcessPoolExecutor") as pool,
              patch("src.neat._training_workers", return_value=2), patch("src.neat.os.cpu_count", return_value=3),
              patch("src.neat.wait", side_effect=wait)):
            pool.return_value.__enter__.return_value.submit.side_effect = submit
            directory = Path(folder)
            with patch("src.neat.REPLAY_DIR", directory / "replays"):
                champion = train(Settings(population=3, seed=13), directory, publish, stop=stop)
            data, genome, _ = read_champion(directory)
            replay = max((directory / "replays").glob("essai-*.json"))
            self.assertEqual(json.loads(replay.read_text(encoding="utf-8")), data["replay"])
        self.assertEqual((data["id"], data["generation"], data["score"], data["time"], genome),
                         (4, 2, 65, 47, champion.genome))

    def test_replay_swaps_pending_champion_at_loop_boundary(self):
        first, second = Replay(empty_match()), Replay(empty_match())
        clock = Playback(first, 1, loop=True)
        clock.speed = 10
        clock.advance(5)
        clock.offer(2, second)
        self.assertEqual((clock.trial, clock.replay.time), (1, 50))
        clock.playing = False
        clock.advance(10)
        self.assertEqual((clock.trial, clock.replay.time), (1, 50))
        clock.playing = True
        clock.advance(5)
        self.assertEqual((clock.trial, clock.replay.time), (1, 100))
        clock.advance(0.01)
        self.assertEqual((clock.trial, clock.replay.time), (2, 0))
        clock.loop = False
        clock.advance(10)
        self.assertFalse(clock.playing)
        clock.restart()
        self.assertEqual(clock.replay.time, 0)

    def test_visual_replay_skips_only_a_stationary_tail_with_the_same_score(self):
        match = empty_match()
        match["frames"].insert(1, dict(match["frames"][0], t=67))
        replay = Replay(match)
        clock = Playback(replay, 1, skip_idle=True)
        clock.advance(100)
        self.assertEqual((replay.time, replay.duration, clock.playing), (67, 100, False))

        final = dict(match["frames"][-1], score=11)
        replay = Replay(dict(match, frames=[*match["frames"][:-1], final]))
        clock = Playback(replay, 2, skip_idle=True)
        clock.advance(100)
        self.assertEqual((replay.time, replay.frame["score"]), (100, 11))


if __name__ == "__main__":
    unittest.main()
