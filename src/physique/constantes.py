"""Constantes physiques provisoires, en attendant les mesures sur le robot réel."""

from math import cos, radians


SPEED = 1000  # mm/s
DRIVE_ACCELERATION = 1000  # mm/s² en translation
TURN_STEP = radians(15)
TURN_SPEED = radians(360)
TURN_ACCELERATION = radians(1200)  # rad/s², effort de pivot provisoire
PHYSICS_STEP = 1 / 480  # contacts plus stables quand une pile est coincée
ROBOT_MASS = 5.0  # kg, masse effective provisoire pour les contacts
PILE_MASS = 0.09
FLOOR_DECELERATION = 4200  # mm/s²
FLOOR_ANGULAR_DECELERATION = 10  # rad/s²
SUCTION_REACH = 5  # mm
SUCTION_ALIGNMENT = cos(radians(20))
