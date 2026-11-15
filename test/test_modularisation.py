import ast
from math import pi
from pathlib import Path
from unittest import TestCase
from unittest.mock import mock_open, patch

import numpy as np

from src.actions import ACTIONS
from src.actions.construction import construction_retenue, peut_construire
from src.composition import composer, profil_defaut, resoudre
from src.reseaux.constante import Constante
from src.reseaux.lineaire import Lineaire
from src.domaine.modele import Candidat, Observation
from src.domaine.robot import CAPACITES, Construction, Mouvement, RobotSpec, Ventouse
from src.domaine.scenario import ScenarioSpec
from src.planification.astar import AstarPlanner
from src.planification.base import DemandeTrajet
from src.planification.direct import DirectPlanner
from src.recompenses.base import Transition
from src.regles.bareme import Bareme2027
from src.recompenses.construction_temps import ConstructionTemps
from src.replay.lecture import Replay
from src.replay.player import LecteurReplay
from src.robots.loader import load_robot_spec
from src.selection.aleatoire import Aleatoire
from src.selection.epsilon import EpsilonGlouton
from src.selection.glouton import Glouton
from src.terrains.officiel_2027 import TERRAIN


class CapabilityTest(TestCase):
    def test_capabilities_are_transitive(self):
        self.assertEqual(CAPACITES[Construction.MUR], {Construction.MUR})
        self.assertEqual(CAPACITES[Construction.TOUR], {Construction.TOUR, Construction.MUR})
        self.assertEqual(CAPACITES[Construction.PORTE],
                         {Construction.PORTE, Construction.TOUR, Construction.MUR})

    def test_bareme_construction_values(self):
        bareme = Bareme2027()
        self.assertEqual(bareme.points_pierre, 1)
        self.assertEqual(bareme.points_construction(Construction.MUR), 5)
        self.assertEqual(bareme.points_construction(Construction.TOUR), 5)
        self.assertEqual(bareme.points_construction(Construction.PORTE), 20)

    def test_robot_spec_filters_actions(self):
        spec = RobotSpec("test", ((0, 0), (10, 0), (0, 10)), (
            Ventouse(1, (0, 0), (1, 0), Construction.TOUR),
        ), Mouvement())
        self.assertTrue(peut_construire(spec, Construction.MUR))
        self.assertTrue(peut_construire(spec, Construction.TOUR))
        self.assertFalse(peut_construire(spec, Construction.PORTE))
        self.assertEqual(construction_retenue(spec, (Construction.MUR, Construction.TOUR)),
                         Construction.TOUR)


class TomlTest(TestCase):
    def test_construction_and_movement_are_read(self):
        config = b"""
nom = "demo"
planificateur = "direct"
contour_mm = [[-100, -50], [100, -50], [100, 50], [-100, 50]]

[mouvement]
type = "differentiel"
vitesse_mm_s = 500
acceleration_mm_s2 = 800
masse_kg = 7

[[ventouses]]
position_mm = [100, 0]
angle_deg = 0
construction = "porte"
"""
        with patch.object(Path, "open", mock_open(read_data=config)):
            spec = load_robot_spec(Path("capacites.toml"))
        self.assertEqual(spec.id, "demo")
        self.assertEqual(spec.planificateur, "direct")
        self.assertEqual(spec.mouvement.type, "differentiel")
        self.assertEqual(spec.mouvement.masse_kg, 7)
        self.assertEqual(spec.ventouses[0].construction, Construction.PORTE)
        self.assertEqual(spec.ventouses[0].capacites,
                         {Construction.PORTE, Construction.TOUR, Construction.MUR})


class PlanningTest(TestCase):
    def test_both_planners_return_a_path_on_a_free_segment(self):
        demande = DemandeTrajet((100, 100), (400, 100), (((0, 0), (10, 0), (10, 10)),), ())
        for planner in (AstarPlanner(), DirectPlanner()):
            trajet = planner.planifier(demande)
            self.assertIsNotNone(trajet, planner.id)
            self.assertGreater(trajet.duree_s, 0)


class RewardSelectionTest(TestCase):
    def test_reward_and_selector_are_swappable(self):
        avant = Observation((0, 0, 0), (), {}, 10, 0.0)
        apres = Observation((0, 0, 0), (), {}, 18, 1.0)
        transition = Transition(avant, Candidat("depose", 8, 1), apres,
                                {"score": 18, "duree": 20})
        self.assertAlmostEqual(ConstructionTemps().valeur(transition),
                               (18 - 10 + 0.5 * 0.8) / 8)
        candidats = [Candidat("a", 2, 2), Candidat("b", 3, 1)]
        self.assertEqual([c.action for c in Glouton().classer(None, candidats)], ["b", "a"])
        self.assertEqual(set(c.action for c in Aleatoire(np.random.default_rng(3)).classer(None, candidats)),
                         {"a", "b"})


class ReplayPlayerTest(TestCase):
    def test_player_moves_without_tkinter(self):
        def frame(t):
            return {"t": t, "position": [0, 0], "heading": 0.0,
                    "robot": [[0, 0], [1, 0], [0, 1]], "piles": [], "cups": [],
                    "target": None, "blocked": False, "score": 0, "action": "test"}
        replay = Replay({"obstacles": [], "frames": [frame(0.0), frame(1.0)]})
        player = LecteurReplay(replay)
        player.avancer(0.5)
        self.assertAlmostEqual(player.temps, 0.5)
        player.chercher(0.9)
        self.assertAlmostEqual(player.temps, 0.9)
        player.chercher(2.0)
        self.assertTrue(player.fin())


class CompositionTest(TestCase):
    def test_all_rewards_and_selectors_are_usable(self):
        avant = Observation((0, 0, 0), (), {}, 10, 0.0)
        apres = Observation((0, 0, 0), (), {}, 18, 1.0)
        transition = Transition(avant, Candidat("depose", 8, 1), apres,
                                {"score": 18, "duree": 20, "score_retour": 5})
        for name in ("construction", "construction_temps", "retour", "depots", "sparse"):
            reward = resoudre("recompense", name)()
            self.assertIsInstance(reward.valeur(transition), float)
            self.assertEqual(len(reward.rang({"score": 10, "duree": 20})), 2)
        candidats = [Candidat("a", 2, 2), Candidat("b", 3, 1)]
        for name in ("glouton",):
            self.assertEqual(len(resoudre("selection", name)().classer(None, candidats)), 2)
        aleatoire = Aleatoire(np.random.default_rng(0))
        self.assertEqual(len(aleatoire.classer(None, candidats)), 2)
        epsilon = EpsilonGlouton(Glouton(), np.random.default_rng(0), 1.0)
        self.assertEqual(len(epsilon.classer(None, candidats)), 2)

    def test_action_catalog_filters_by_capability(self):
        composition = composer(profil_defaut())
        noms = {action.id for action in composition.actions}
        self.assertIn("prise", noms)
        self.assertIn("depose", noms)
        self.assertIn("retour", noms)
        self.assertIn("construire_mur", noms)
        self.assertNotIn("construire_tour", noms)
        self.assertNotIn("construire_porte", noms)
        self.assertTrue(any(action.id == "construire_tour" for action in ACTIONS))

    def test_second_robot_is_available(self):
        robot = resoudre("robot", "rectangle_demo")()
        self.assertEqual(robot.planificateur, "direct")
        self.assertEqual(robot.ventouses[0].construction, Construction.TOUR)
        principal = resoudre("robot", "robotronik")()
        scenario = ScenarioSpec(TERRAIN, principal, adversaires=(robot,))
        self.assertEqual(scenario.adversaires[0].id, "rectangle_demo")

    def test_network_variants_are_registered(self):
        self.assertIs(resoudre("reseau", "lineaire"), Lineaire)
        self.assertIs(resoudre("reseau", "constante"), Constante)
        lineaire = Lineaire(np.random.default_rng(0))
        constante = Constante(2.0)
        self.assertEqual(constante.predict(np.zeros((2, 97), dtype=np.float32)).tolist(), [2.0, 2.0])
        self.assertEqual(lineaire.predict(np.zeros((2, 97), dtype=np.float32)).shape, (2,))

    def test_default_composition_resolves(self):
        composition = composer(profil_defaut())
        self.assertEqual(composition.planificateur.id, "astar")
        self.assertEqual(composition.recompense.id, "construction_temps")
        self.assertEqual(composition.selecteur.id, "glouton")
        self.assertIs(resoudre("terrain", "officiel_2027")(), TERRAIN)
        self.assertEqual(resoudre("planification", "direct")().id, "direct")


class IsolationTest(TestCase):
    def test_domain_has_no_heavy_imports(self):
        forbidden = {"pymunk", "numpy", "tkinter", "PIL"}
        for path in Path("src/domaine").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = {(node.module or "").split(".")[0]}
                else:
                    continue
                self.assertFalse(names & forbidden, f"{path} importe {names & forbidden}")

    def test_engine_does_not_import_loader_or_ui_at_module_level(self):
        tree = ast.parse(Path("src/physique/moteur.py").read_text(encoding="utf-8"))
        modules = {node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
        self.assertFalse(any("robots.loader" in module or "interface" in module for module in modules))
