"""État de lecture d'un replay, indépendant de Tkinter."""

from time import perf_counter

from ..replay.player import LecteurReplay


class ControleurReplay:
    def __init__(self, replay, *, horloge=perf_counter):
        self.lecteur = LecteurReplay(replay)
        self.horloge = horloge
        self.joue = True
        self.derniere = horloge()

    def basculer(self):
        self.joue = not self.joue
        self.derniere = self.horloge()
        if self.joue and self.lecteur.fin():
            self.lecteur.debut()

    def chercher(self, temps: float):
        self.lecteur.chercher(temps)

    def avancer(self):
        maintenant = self.horloge()
        if self.joue:
            self.lecteur.avancer(maintenant - self.derniere)
        self.derniere = maintenant
