"""Freinage des piles sur le tapis."""

import pymunk


def ajouter_freinage_sol(space, static, body, moment, *, masse, deceleration, angular_deceleration):
    """Ajoute les joints qui freinent le glissement et la rotation sans ancrer la pile."""
    slide = pymunk.PivotJoint(static, body, (0, 0), (0, 0))
    spin = pymunk.GearJoint(static, body, 0, 1)
    slide.max_bias = spin.max_bias = 0
    slide.max_force = masse * deceleration
    spin.max_force = moment * angular_deceleration
    return slide, spin
