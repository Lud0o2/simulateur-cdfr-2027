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


class _Construction:
    construction = None

    def candidats(self, ctx):
        return ()

    def peut_executer(self, ctx) -> bool:
        return self.construction is not None and ctx.robot.spec.supports(self.construction)

    def executer(self, ctx, candidat) -> bool:
        if self.construction is None or not self.peut_executer(ctx):
            return False
        payload = candidat.payload
        from .depose import place_pile

        return place_pile(
            ctx.moteur, payload["cup"], payload["x"], payload["y"], payload["angle"],
            deadline=ctx.deadline, planner=ctx.planificateur,
        )


class ConstructionMur(_Construction):
    id = "construire_mur"
    construction = Construction.MUR


class ConstructionTour(_Construction):
    id = "construire_tour"
    construction = Construction.TOUR


class ConstructionPorte(_Construction):
    id = "construire_porte"
    construction = Construction.PORTE
