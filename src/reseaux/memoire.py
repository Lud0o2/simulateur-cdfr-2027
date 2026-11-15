"""Mémoire d'expérience pour l'entraînement."""

from collections import OrderedDict


class MemoireExperiences:
    def __init__(self, capacite: int | None = None):
        self.capacite = capacite
        self._valeurs = OrderedDict()

    def ajouter(self, cle, valeur):
        self._valeurs[cle] = valeur
        if self.capacite is not None and len(self._valeurs) > self.capacite:
            self._valeurs.popitem(last=False)

    def obtenir(self, cle, defaut=None):
        return self._valeurs.get(cle, defaut)

    def valeurs(self):
        return tuple(self._valeurs.values())

    def __len__(self):
        return len(self._valeurs)
