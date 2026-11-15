"""Capacités de construction d'un robot."""

from ..domaine.robot import Construction, RobotSpec


PRIORITE = {Construction.MUR: 1, Construction.TOUR: 2, Construction.PORTE: 3}


def peut_construire(spec: RobotSpec, construction: Construction) -> bool:
    return spec.supports(construction)


def priorite(construction: Construction) -> int:
    return PRIORITE[construction]


def construction_retenue(spec: RobotSpec, constructions) -> Construction | None:
    """Retient la construction la plus prioritaire autorisée par le robot."""
    autorisees = [construction for construction in constructions if peut_construire(spec, construction)]
    return max(autorisees, key=priorite, default=None)
