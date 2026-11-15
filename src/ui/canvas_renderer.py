"""Calculs d'état pour le rendu, sans Tkinter."""


def etat_robot(simulation):
    move = simulation.movement
    moving = simulation.moving
    target = move.end if moving and not move.turn else None
    blocked = move is not None and (move.blocked or move.rotation_blocked)
    if target is not None:
        label = "En déplacement"
    elif moving:
        label = "En rotation"
    elif move is None:
        label = "Au départ"
    elif move.rotation_blocked:
        label = "Rotation bloquée"
    elif move.turn:
        label = "Rotation bloquée" if blocked else "Cap réglé"
    else:
        label = "Trajet bloqué" if blocked else "Arrivé"
    return target, blocked, label
