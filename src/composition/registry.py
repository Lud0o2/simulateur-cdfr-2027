"""Registre des implémentations interchangeables."""

from importlib import import_module


FABRIQUES = {
    "robot": {"robotronik": "src.robots.robotronik:construire"},
    "terrain": {"officiel_2027": "src.terrains.officiel_2027:construire"},
    "scenario": {"officiel_2027": "src.evaluation.scenarios:officiel_2027"},
    "planification": {
        "astar": "src.planification.astar:AstarPlanner",
        "direct": "src.planification.direct:DirectPlanner",
    },
    "recompense": {
        "construction": "src.recompenses.construction:Construction",
        "construction_temps": "src.recompenses.construction_temps:ConstructionTemps",
        "retour": "src.recompenses.retour:Retour",
        "depots": "src.recompenses.depots:Depots",
        "sparse": "src.recompenses.sparse:Sparse",
    },
    "selection": {
        "glouton": "src.selection.glouton:Glouton",
        "aleatoire": "src.selection.aleatoire:Aleatoire",
        "reseau": "src.selection.reseau:MatchPolicy",
    },
    "reseau": {
        "mlp": "src.reseaux.mlp:Network",
        "lineaire": "src.reseaux.lineaire:Lineaire",
        "constante": "src.reseaux.constante:Constante",
    },
}


def resoudre(categorie: str, identifiant: str):
    chemin = FABRIQUES[categorie][identifiant]
    module, attribut = chemin.split(":")
    return getattr(import_module(module), attribut)
