"""Horloge réelle ou factice pour l'interface, le reporting et les tests."""

from time import monotonic, perf_counter


class Horloge:
    def maintenant(self) -> float:
        return perf_counter()


class HorlogeMonotone(Horloge):
    def maintenant(self) -> float:
        return monotonic()


class HorlogeFactice(Horloge):
    def __init__(self, temps: float = 0.0):
        self.temps = temps

    def maintenant(self) -> float:
        return self.temps

    def avancer(self, secondes: float) -> None:
        self.temps += secondes
