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


class Playback:
    """Lecture d'un replay avec boucle, vitesse et remplacement du replay affiché."""

    def __init__(self, replay: Replay, trial, *, loop: bool = False, skip_idle: bool = False):
        self.lecteur = LecteurReplay(replay)
        self.trial = trial
        self.loop = loop
        self.skip_idle = skip_idle
        self.end = self._end_time()
        self.playing = True
        self.speed = 1.0
        self.pending = None

    @property
    def replay(self) -> Replay:
        return self.lecteur.replay

    def _end_time(self) -> float:
        frames = self.replay.frames
        if self.skip_idle and len(frames) > 1:
            previous, last = frames[-2:]
            # Le buzzer reste dans le fichier. On peut sauter l'attente si l'état affiché est identique.
            if all(previous.get(key) == value for key, value in last.items() if key not in {"t", "action"}):
                return previous["t"]
        return self.replay.duration

    def offer(self, trial, replay: Replay):
        self.pending = trial, replay
        if self.trial is None:
            self.restart()

    def restart(self):
        if self.pending is not None:
            self.trial, replay = self.pending
            self.lecteur = LecteurReplay(replay)
            self.pending = None
            self.end = self._end_time()
        self.lecteur.debut()

    def avancer(self, secondes: float) -> bool:
        if not self.playing or (self.replay.start == self.end and self.pending is None):
            return False
        if self.loop and self.lecteur.temps >= self.end:
            self.restart()
            return True
        self.lecteur.chercher(min(self.end, self.lecteur.temps + secondes * self.speed))
        if not self.loop and self.lecteur.temps >= self.end:
            self.playing = False
        return True
