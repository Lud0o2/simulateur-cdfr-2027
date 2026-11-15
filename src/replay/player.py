"""Lecteur de replay sans dépendance à Tkinter."""

from .lecture import Replay


class LecteurReplay:
    def __init__(self, replay: Replay):
        self.replay = replay

    @property
    def temps(self) -> float:
        return self.replay.time

    @property
    def frame(self):
        return self.replay.frame

    def avancer(self, secondes: float) -> None:
        self.replay.time = min(self.replay.duration, self.replay.time + max(0.0, secondes))

    def chercher(self, temps: float) -> None:
        self.replay.time = max(self.replay.start, min(self.replay.duration, temps))

    def debut(self) -> None:
        self.replay.time = self.replay.start

    def fin(self) -> bool:
        return self.replay.time >= self.replay.duration
