"""Contrôleur Tkinter de lecture d'un replay."""

from time import perf_counter

from ..replay.player import Playback

__all__ = ["ControleurReplayTk", "Playback"]


class ControleurReplayTk:
    """Barre de lecture, bouton, curseur et options d'un replay."""

    def __init__(self, root, footer, playback, draw, *, theme, tk, ttk, dashboard=None):
        self.root = root
        self.playback = playback
        self.draw = draw
        self.dashboard = dashboard
        self.glisse = False
        self.derniere = perf_counter()
        self.timers = {}

        self.controls = tk.Frame(footer, bg=theme.surface)
        self.controls.pack(fill="x")
        self.button = ttk.Button(self.controls, command=self.basculer, width=10, style="Panel.TButton")
        self.button.pack(side="left", padx=(0, 10))
        if dashboard is not None:
            ttk.Button(self.controls, text="Recommencer", command=self.recommencer,
                       style="Panel.TButton").pack(side="left", padx=(0, 10))
        replay = playback.replay
        self.timeline = ttk.Scale(
            self.controls, from_=replay.start, to=playback.end, value=replay.time,
            command=self.chercher, style="Replay.Horizontal.TScale", cursor="hand2",
        )
        self.timeline.pack(side="left", expand=True, fill="x")
        self.timeline.bind("<Button-1>", self.glisser)
        self.timeline.bind("<B1-Motion>", self.glisser)
        self.timeline.bind("<ButtonRelease-1>", self.relacher)
        if dashboard is not None:
            self.boucle = tk.BooleanVar(master=root, value=playback.loop)
            self.vitesse = tk.StringVar(master=root, value="1")
            ttk.Style(root).configure(
                "Replay.TCheckbutton", background=theme.surface,
                foreground=theme.foreground, font=(theme.font, 9),
            )
            ttk.Checkbutton(
                self.controls, text="Boucle", variable=self.boucle, style="Replay.TCheckbutton",
                command=lambda: setattr(playback, "loop", self.boucle.get()),
            ).pack(side="left", padx=8)
            tk.Label(self.controls, text="×", bg=theme.surface, fg=theme.foreground,
                     font=(theme.font, 10)).pack(side="left")
            speeds = ttk.Combobox(self.controls, textvariable=self.vitesse, state="readonly", width=4,
                                  values=("0,5", "1", "2", "5", "10"))
            speeds.pack(side="left", padx=(2, 0))
            speeds.bind("<<ComboboxSelected>>", self.regler_vitesse)
            self._planifier(150, self.sonder)
        self.root.bind("<Destroy>", self.annuler)
        self.rafraichir()
        self._planifier(16, self.tick)

    def _planifier(self, delay, action):
        self.timers[action] = self.root.after(delay, action)

    def annuler(self, event):
        if event.widget is self.root:
            for timer in self.timers.values():
                self.root.after_cancel(timer)
            self.timers.clear()

    def rafraichir(self):
        playback = self.playback
        fini = playback.lecteur.temps >= playback.end
        if fini and not playback.loop and not self.glisse:
            playback.playing = False
        self.button.configure(text="Pause" if playback.playing else "Relancer" if fini else "Reprendre")
        self.draw()

    def synchroniser(self):
        replay = self.playback.replay
        self.timeline.configure(from_=replay.start, to=self.playback.end)
        self.timeline.set(replay.time)
        self.rafraichir()

    def recommencer(self):
        self.playback.restart()
        self.playback.playing = True
        self.derniere = perf_counter()
        self.synchroniser()

    def basculer(self):
        playback = self.playback
        playback.playing = not playback.playing
        self.derniere = perf_counter()
        if playback.playing and playback.lecteur.temps >= playback.end:
            self.recommencer()
        else:
            self.rafraichir()

    def chercher(self, value):
        self.playback.lecteur.chercher(float(value))
        self.rafraichir()

    def glisser(self, event):
        self.glisse = True
        self.timeline.focus_set()
        self.timeline.set(self.timeline.get(event.x, event.y))
        return "break"

    def relacher(self, _event):
        self.glisse = False
        self.derniere = perf_counter()
        self.rafraichir()

    def regler_vitesse(self, _event):
        self.playback.speed = float(self.vitesse.get().replace(",", "."))

    def tick(self):
        maintenant = perf_counter()
        if not self.glisse and self.playback.avancer(maintenant - self.derniere):
            if self.playback.end != float(self.timeline.cget("to")):
                self.synchroniser()
            else:
                self.timeline.set(self.playback.lecteur.temps)
        self.derniere = maintenant
        self._planifier(16, self.tick)

    def sonder(self):
        incoming = self.dashboard.poll()
        if incoming is not None:
            self.playback.offer(*incoming)
            self.synchroniser()
        self._planifier(150, self.sonder)
