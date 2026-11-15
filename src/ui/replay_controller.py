"""Contrôleur Tkinter de lecture d'un replay."""

from time import perf_counter

from ..replay.player import LecteurReplay


class ControleurReplay:
    """État de lecture indépendant de Tkinter."""

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


class ControleurReplayTk:
    """Barre de lecture, bouton et curseur d'un replay."""

    def __init__(self, root, footer, replay, surface, draw, *, tk, ttk):
        self.root = root
        self.replay = replay
        self.draw = draw
        self.controleur = ControleurReplay(replay)
        self.glisse = False
        self.controls = tk.Frame(footer, bg=surface)
        self.controls.pack(fill="x")
        self.button = ttk.Button(self.controls, command=self.basculer, width=10, style="Panel.TButton")
        self.button.pack(side="left", padx=(0, 10))
        self.timeline = ttk.Scale(
            self.controls, from_=replay.start, to=replay.duration, value=replay.time,
            command=self.chercher, style="Replay.Horizontal.TScale", cursor="hand2",
        )
        self.timeline.pack(side="left", expand=True, fill="x")
        self.timeline.bind("<Button-1>", self.glisser)
        self.timeline.bind("<B1-Motion>", self.glisser)
        self.timeline.bind("<ButtonRelease-1>", self.relacher)
        self.rafraichir()
        self.tick()

    def rafraichir(self):
        fini = self.replay.time >= self.replay.duration
        if fini and not self.glisse:
            self.controleur.joue = False
        self.button.configure(text="Pause" if self.controleur.joue else "Relancer" if fini else "Reprendre")
        self.draw()

    def basculer(self):
        fini = self.replay.time >= self.replay.duration
        self.controleur.basculer()
        if self.controleur.joue and fini:
            self.timeline.set(self.replay.start)
        else:
            self.rafraichir()

    def chercher(self, value):
        self.controleur.chercher(float(value))
        self.rafraichir()

    def glisser(self, event):
        self.glisse = True
        self.timeline.focus_set()
        self.timeline.set(self.timeline.get(event.x, event.y))
        return "break"

    def relacher(self, _event):
        self.glisse = False
        self.controleur.derniere = perf_counter()
        self.rafraichir()

    def tick(self):
        if not self.glisse:
            self.controleur.avancer()
            self.timeline.set(self.replay.time)
        self.root.after(16, self.tick)
