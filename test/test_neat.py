from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from src.interface import Playback
from src.match import MATCH_SECONDS, Replay, play_match
from src.neat import (CHAMPION_FILE, MAX_HIDDEN, MAX_LINKS, N_INPUTS, Genome, Individual, Settings, fitness,
                      initial_genome, mutate, offspring, parent_weights, read_champion, train, write_champion)
from src.simulateur import Simulation, WALL_ZONES


def empty_match():
    return play_match(Simulation(blocks=()), full_match=True)


class FitnessTest(unittest.TestCase):
    def test_points_always_precede_time(self):
        matches = [(60, 30), (70, 95), (65, 50), (70, 90)]
        ranking = sorted(matches, key=lambda pair: fitness(dict(score_max=pair[0], temps_score_max=pair[1])),
                         reverse=True)
        self.assertEqual(ranking, [(70, 90), (70, 95), (65, 50), (60, 30)])

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

    def test_parent_selection_favors_rank_and_keeps_elites(self):
        settings = Settings(population=4)
        genome = initial_genome(Random(8), 0)
        ranked = [Individual(i, 1, genome, rank) for i, rank in enumerate(((65, -90), (65, -90), (60, -10), (55, -5)))]
        self.assertEqual(parent_weights(ranked), [4, 4, 2, 1])
        children = offspring(ranked, Random(17), settings, 0.3, 2, 5)
        self.assertEqual(len(children), settings.population)
        self.assertIs(children[0], ranked[0])
        self.assertTrue(all(child.rank is None for child in children[1:]))
        self.assertEqual(children, offspring(ranked, Random(17), settings, 0.3, 2, 5))


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

    def test_generations_keep_champion_and_reuse_elite_evaluation(self):
        events = []
        pause = Event()
        pause.set()
        match = empty_match()

        def publish(event):
            events.append(event)
            if event["status"] == "Pause":
                self.assertEqual(event["simulations"], 0)
                pause.clear()

        def evaluate(_genome, record_frames=False):
            return match if record_frames else fitness(match)

        with (TemporaryDirectory() as folder, patch("src.neat.ProcessPoolExecutor", ThreadPoolExecutor),
              patch("src.neat.evaluate", side_effect=evaluate)):
            directory = Path(folder)
            train(Settings(population=2, generations=2, seed=13), directory, publish, pause=pause)
            history = [json.loads(line) for line in (directory / "generations.jsonl").read_text().splitlines()]
            data, _, _ = read_champion(directory)
        self.assertEqual([stats["simulations"] for stats in history], [2, 3])
        self.assertEqual(data["generation"], 1)
        self.assertEqual(data["id"], 1)
        self.assertEqual(events[-1]["status"], "Terminé")
        self.assertIn("Pause", [event["status"] for event in events])

    def test_next_generation_starts_during_replay_and_stop_saves_latest_champion(self):
        match = empty_match()
        stop = Event()
        scores = {}
        first_recording = None

        def submit(_evaluate, genome, record_frames=False):
            nonlocal first_recording
            future = Future()
            if record_frames:
                score = scores[genome]
                replay = dict(match, score=score, score_max=score, frames=[dict(frame, score=score) for frame in match["frames"]])
                if first_recording is None:
                    first_recording = future, replay
                else:
                    future.set_result(replay)
            else:
                score = scores[genome] = 11 + len(scores)
                if len(scores) == 4:
                    pending, replay = first_recording
                    self.assertFalse(pending.done())
                    pending.set_result(replay)
                future.set_result((score, 0))
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
            champion = train(Settings(population=3, seed=13), directory, publish, stop=stop)
            data, genome, _ = read_champion(directory)
        self.assertEqual((data["id"], data["generation"], data["score"], genome), (4, 2, 14, champion.genome))

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
