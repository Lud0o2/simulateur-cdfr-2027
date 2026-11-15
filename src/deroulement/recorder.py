"""Politique d'enregistrement des frames."""

from ..replay.recorder import _frame


class Enregistreur:
    def __init__(self, actif: bool = True):
        self.actif = actif
        self.frames = []

    def frame(self, simulation, label: str, bonus: float | None = None):
        if not self.actif:
            return
        frame = _frame(simulation, label)
        frame["score"] += bonus or 0
        if self.frames and self.frames[-1]["t"] == frame["t"]:
            self.frames[-1] = frame
        else:
            self.frames.append(frame)
