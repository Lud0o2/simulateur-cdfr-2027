import unittest
from math import dist, radians

from simulateur import FOOTPRINT, SIDE, SPEED, START, TURN_STEP, Simulation


class SimulationTest(unittest.TestCase):
    def test_translation_holonome_and_time(self):
        sim = Simulation()
        sim.move_to(250, 2000 - SIDE / 2)
        sim.advance(0.5)
        self.assertEqual(sim.position, (250, 1500))
        sim.advance(0.5)
        self.assertEqual(sim.position, (250, 2000 - SIDE / 2))
        self.assertFalse(sim.moving)
        for i, vertex in enumerate(FOOTPRINT):
            self.assertAlmostEqual(dist(FOOTPRINT[i - 1], vertex), SIDE)

    def test_entire_triangle_stays_inside_table_and_can_leave_the_border(self):
        sim = Simulation()
        sim.move_to(1500, 0)
        sim.advance(10)
        self.assertTrue(sim.movement.blocked)
        self.assertAlmostEqual(sim.position[1], SIDE / 2, places=5)
        self.assertLess(sim.position[0], 1500)
        sim.move_to(1500, 1000)
        sim.advance(10)
        self.assertEqual(sim.position, (1500, 1000))

    def test_crossing_an_obstacle_stops_even_with_a_single_large_time_step(self):
        sim = Simulation(((800, 1150, 850, 1200),))
        sim.move_to(1300, 1000)
        sim.advance(10)
        # Le centre passe à côté du rectangle, mais le coin avant le touche.
        contact_x = 800 - max(x for x, _ in FOOTPRINT)
        self.assertTrue(sim.movement.blocked)
        self.assertAlmostEqual(sim.position[0], contact_x, places=5)
        self.assertAlmostEqual(sim.movement.duration, (contact_x - START[0]) / SPEED, places=6)
        sim.move_to(*START)
        sim.advance(10)
        self.assertEqual(sim.position, START)

    def test_an_obstacle_can_fit_inside_the_bounding_box_beside_the_triangle(self):
        sim = Simulation(((1000, 1170, 1020, 1190),))
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
        sim = Simulation()
        sim.move_to(1250, 1000)
        sim.advance(0.5)
        sim.move_to(750, 500)
        sim.advance(0.5)
        self.assertEqual(sim.position, (750, 500))
        sim.reset()
        self.assertEqual(sim.position, START)
        self.assertEqual(sim.time, 0)

    def test_turn_preserves_center_and_updates_collision_footprint(self):
        sim = Simulation()
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
        sim = Simulation()
        sim.turn_by(radians(-37.5))
        sim.move_to(250, 219)
        sim.advance(1)
        self.assertFalse(sim.movement.blocked)
        sim.turn_by(TURN_STEP)
        self.assertTrue(sim.movement.blocked)
        self.assertAlmostEqual(sim.heading, radians(-37.5))

    def test_turn_cannot_sweep_through_an_obstacle_with_clear_endpoints(self):
        sim = Simulation(((31.5, 970.5, 33.5, 972.5),))
        sim.turn_by(TURN_STEP)
        self.assertTrue(sim.movement.blocked)
        self.assertEqual(sim.heading, 0)
        sim.turn_by(-TURN_STEP)
        self.assertFalse(sim.movement.blocked)
        self.assertAlmostEqual(sim.heading, -TURN_STEP)


if __name__ == "__main__":
    unittest.main()
