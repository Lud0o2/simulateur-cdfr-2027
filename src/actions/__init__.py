"""Actions interchangeables."""

from .base import Action, Candidat, ContexteAction
from .construction import ConstructionMur, ConstructionPorte, ConstructionTour
from .depose import Depose, place_pile
from .prise import Prise, take_pile
from .retour import Retour

ACTIONS = (Prise(), Depose(), Retour(), ConstructionMur(), ConstructionTour(), ConstructionPorte())

__all__ = [
    "Action", "Candidat", "ContexteAction", "Prise", "Depose", "Retour",
    "ConstructionMur", "ConstructionTour", "ConstructionPorte", "ACTIONS",
    "take_pile", "place_pile",
]
