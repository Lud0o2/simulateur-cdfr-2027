"""Façade de compatibilité de l'ancien module `simulateur`.

Le moteur réel vit dans `src.physique.moteur`. Les nouveaux modules doivent
importer directement les paquets `domaine`, `physique`, `regles`, `robots` et
`terrains`. Cette façade conserve les imports historiques des tests, de
`match.py`, de `reseau.py` et de `interface.py`.
"""

from .domaine.constantes import CONTACT_GAP
from .domaine.geometrie import (
    _footprint_radius, _polygon_bounds, _rotated_vertices, arc_hits_edge, contact_axes,
    edges, polygon_contact, rectangle_corners, swept_contact, travel_time,
)
from .domaine.modele import Block, Movement, Point, Rectangle, SuctionCup
from .domaine.validation import _finite_number, _read_point
from .physique.constantes import (
    DRIVE_ACCELERATION, FLOOR_ANGULAR_DECELERATION, FLOOR_DECELERATION, PHYSICS_STEP,
    PILE_MASS, ROBOT_MASS, SPEED, SUCTION_ALIGNMENT, SUCTION_REACH, TURN_ACCELERATION,
    TURN_SPEED, TURN_STEP,
)
from .physique.moteur import Simulation
from .regles.score_2027 import (
    construction_score, free_wall_slots, scoring_zones, table_contact, throne_room_score,
)
from .robots.loader import ROBOT_FILE, load_robot, load_robot_spec
from .robots.spec import RobotGeometry
from .terrains.officiel_2027 import (
    START, START_BLOCKS, TABLE, TABLE_HEIGHT, TABLE_WIDTH, THRONE_ROOM, TOWER_CENTERS,
    TOWER_RADIUS, WALL_ZONES,
)

__all__ = [
    "CONTACT_GAP", "DRIVE_ACCELERATION", "FLOOR_ANGULAR_DECELERATION", "FLOOR_DECELERATION",
    "PHYSICS_STEP", "PILE_MASS", "ROBOT_MASS", "SPEED", "SUCTION_ALIGNMENT", "SUCTION_REACH",
    "TURN_ACCELERATION", "TURN_SPEED", "TURN_STEP", "ROBOT_FILE", "START", "START_BLOCKS",
    "TABLE", "TABLE_HEIGHT", "TABLE_WIDTH", "THRONE_ROOM", "TOWER_CENTERS", "TOWER_RADIUS",
    "WALL_ZONES", "Block", "Movement", "Point", "Rectangle", "SuctionCup", "Simulation",
    "RobotGeometry", "load_robot", "load_robot_spec", "construction_score", "free_wall_slots",
    "scoring_zones", "table_contact", "throne_room_score", "_finite_number", "_read_point",
    "_polygon_bounds", "_rotated_vertices", "_footprint_radius", "arc_hits_edge",
    "contact_axes", "edges", "polygon_contact", "rectangle_corners", "swept_contact",
    "travel_time",
]
