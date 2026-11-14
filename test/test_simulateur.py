import json
import unittest
from functools import partial
from math import cos, dist, radians, sin, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import mock_open, patch

from pymunk import Vec2d

from src.match import MATCH_SECONDS, Replay, play_match, save_replay
from src.simulateur import START, TURN_STEP, RobotGeometry, Simulation, load_robot


# La géométrie de référence garde les scénarios reproductibles après édition du TOML.
SIDE = 380
FRONT = SIDE * sqrt(3) / 6
FOOTPRINT = ((-2 * FRONT, 0), (FRONT, -SIDE / 2), (FRONT, SIDE / 2))
reference_simulation = partial(Simulation, robot=RobotGeometry(FOOTPRINT, (
    ((FRONT, 0), Vec2d(1, 0)),
    ((-FRONT / 2, SIDE / 4), Vec2d(1, 0).rotated(radians(120))),
    ((-FRONT / 2, -SIDE / 4), Vec2d(1, 0).rotated(radians(-120))),
)))


class SimulationTest(unittest.TestCase):
    def test_translation_holonome_and_time(self):
        sim = reference_simulation()
        sim.move_to(250, 2000 - SIDE / 2)
        sim.advance(0.5)
        self.assertAlmostEqual(dist(sim.position, (250, 1500)), 0)
        sim.advance(0.5)
        self.assertEqual(sim.position, (250, 2000 - SIDE / 2))
        self.assertFalse(sim.moving)
        for i, vertex in enumerate(FOOTPRINT):
            self.assertAlmostEqual(dist(FOOTPRINT[i - 1], vertex), SIDE)

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
        sim.advance(0.5)
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
        sim.advance(1)
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
        sim.advance(0.35)
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
        sim.advance(1)
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
        sim.advance(1.25)
        sim.move_to(1500, 0)
        sim.advance(0.4)
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
                sim.advance(1)
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
        robot = RobotGeometry(FOOTPRINT, (
            ((FRONT, -100), Vec2d(1, 0)),
            ((FRONT, 100), Vec2d(1, 0)),
        ))
        sim = Simulation(blocks=((1445, 440, 1555, 760), (1445, 1040, 1555, 1360)), robot=robot)
        pickup_x = 1445 - FRONT - 2

        sim.move_to(250, 800)
        sim.advance(1)
        sim.move_to(pickup_x, 800)
        sim.advance(2)
        sim.toggle_suction(1)
        self.assertEqual(sim.held_pile(1), 0)

        sim.move_to(pickup_x, 1000)
        sim.advance(1)
        sim.toggle_suction(2)
        self.assertEqual((sim.held_pile(1), sim.held_pile(2)), (0, 1))

        # Place le centre de la seconde pile dans le rempart, avec les deux piles encore tenues.
        sim.move_to(pickup_x - (1500 - 850), 1000)
        sim.advance(1)
        self.assertAlmostEqual(dist(sim.blocks[1].position, (850, 1200)), 0)
        self.assertEqual(sim.score, 0)
        sim.toggle_suction(2)
        self.assertEqual((sim.held_pile(1), sim.held_pile(2)), (0, None))
        self.assertEqual(sim.score, 8)

        first_offset = Vec2d(*sim.blocks[0].position) - Vec2d(*sim.position)
        deposited = sim.blocks[1]
        sim.move_to(*START)
        sim.advance(1)
        self.assertEqual(sim.position, START)
        self.assertEqual((sim.held_pile(1), sim.held_pile(2)), (0, None))
        self.assertAlmostEqual(dist(sim.blocks[0].position, Vec2d(*START) + first_offset), 0)
        self.assertEqual(sim.blocks[1], deposited)
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
        sim.advance(1)
        sim.turn_by(radians(-80))
        x, y = sim.position
        px, py = sim.blocks[0].position
        sim.move_to(480 + x - px, 240 + y - py)
        sim.advance(1)
        sim.toggle_suction(1)
        self.assertEqual(sim.score, 3)


class AutomaticMatchTest(unittest.TestCase):
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
            self.assertEqual(Replay(json.loads(first.read_text(encoding="utf-8"))).duration, MATCH_SECONDS)
            self.assertEqual(save_replay(replay, directory)[0], 2)

    def test_replay_starts_at_first_timestamp(self):
        simulation = reference_simulation(blocks=())
        simulation.advance(5)
        replay = Replay(play_match(simulation))
        self.assertEqual(replay.time, 5)
        self.assertEqual(replay.frame["t"], 5)
        replay.time = 0
        self.assertEqual(replay.frame["t"], 5)

    def test_replay_rejects_a_later_frame_with_a_missing_field(self):
        result = play_match(reference_simulation(blocks=()))
        del result["frames"][-1]["target"]
        with self.assertRaisesRegex(ValueError, "image 2 .*target.* manquant"):
            Replay(result)

    def test_plays_one_delivery_and_reopens_numbered_attempts(self):
        simulation = reference_simulation(blocks=((1445, 340, 1555, 660),))
        result = play_match(simulation)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            first = save_replay(result, Path(folder))
            second = save_replay(result, Path(folder))
            self.assertEqual((first[0], second[0]), (1, 2))
            self.assertEqual(second[1].name, "essai-0002.json")
            replay = Replay(json.loads(first[1].read_text(encoding="utf-8")))
        self.assertEqual(replay.frame["score"], 0)
        self.assertTrue(any(frame["action"].startswith("Prise") for frame in replay.frames))
        self.assertTrue(any(frame["action"].startswith("Dépôt") for frame in replay.frames))
        self.assertEqual(replay.times, sorted(set(replay.times)))
        self.assertEqual(replay.duration, MATCH_SECONDS)
        replay.time = MATCH_SECONDS
        self.assertEqual(replay.frame["score"], 8)
        self.assertEqual(simulation.score, 8)


class RobotConfigurationTest(unittest.TestCase):
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
