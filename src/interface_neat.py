"""Tableau de bord NEAT et replay du champion dans la fenêtre du simulateur."""

from collections import deque
import json
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Thread
from time import perf_counter
import tkinter as tk
from tkinter import ttk

from .interface import ACCENT, BACKGROUND, BORDER_COLOR, FONT, FOREGROUND, MUTED, SURFACE, WARNING, YELLOW, create_window
from .match import Replay, _frame
from .neat import Genome, INPUTS, N_INPUTS, new_directory, train
from .simulateur import Simulation


NEGATIVE = "#e49384"


def _draw_network(canvas, genome):
    canvas.delete("all")
    width, height = canvas.winfo_width(), canvas.winfo_height()
    if width <= 1 or height <= 1:
        return
    if genome is None:
        canvas.create_text(width / 2, height / 2, text="En attente du premier réseau", fill=MUTED, font=(FONT, 9))
        return
    layers = [0] * (genome.output + 1)
    for source, target, _ in sorted(genome.links, key=lambda edge: edge[1]):
        layers[target] = max(layers[target], layers[source] + 1)
    layers[-1] = max(layers[:-1]) + 1
    columns = [[] for _ in range(layers[-1] + 1)]
    for node, depth in enumerate(layers):
        columns[depth].append(node)
    compact = width < 460
    left, right = (94, width - 44) if compact else (142, width - 100)
    spacing = (right - left) / layers[-1]
    coordinates = {}
    for depth, nodes in enumerate(columns):
        x = left + depth * spacing
        for index, node in enumerate(nodes, 1):
            coordinates[node] = x, 25 + (height - 37) * index / (len(nodes) + 1)
    links = genome.links
    if compact and len(links) > 45:
        links = sorted(links, key=lambda edge: abs(edge[2]), reverse=True)[:45]
    for source, target, weight in links:
        color = ACCENT if weight > 0 else NEGATIVE
        canvas.create_line(*coordinates[source], *coordinates[target], fill=color if weight else BORDER_COLOR,
                           width=min(3.4, 0.7 + abs(weight) * 1.1))
    for node, (x, y) in coordinates.items():
        if node < N_INPUTS:
            radius, color = 3, MUTED
            canvas.create_text(x - 8, y, anchor="e", text=INPUTS[node], font=(FONT, 7 if compact else 10), fill=FOREGROUND)
        elif node == genome.output:
            radius, color = 5, YELLOW
            canvas.create_text(x + 9, y, anchor="w", text="Choix", font=(FONT, 8 if compact else 11), fill=YELLOW)
        else:
            radius, color = 4, ACCENT
            if not compact:
                canvas.create_text(x, y + 12, text=str(node - N_INPUTS + 1), font=(FONT, 9), fill=MUTED)
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=SURFACE)
    for x, anchor, label in ((10, "w", "Entrées"), (width / 2, "center", "Cachés"), (width - 9, "e", "Sortie")):
        if label == "Cachés" and len(genome.biases) == 1:
            continue
        canvas.create_text(x, 12, anchor=anchor, text=label, font=(FONT, 8, "bold"), fill=MUTED)


class Dashboard:
    def __init__(self, settings, directory, saved, initial=None):
        self.settings = settings
        self.directory = directory
        self.initial = initial
        self.events = SimpleQueue()
        self.stop, self.pause = Event(), Event()
        self.progress = {}
        self.stats = []
        self.genome = None
        self.zoom = None
        self.started = perf_counter()
        if saved is not None:
            data, replay = saved
            self.events.put(dict(champion=data, replay=replay, status="Replay enregistré",
                                 log=f"Champion #{data['id']} chargé"))
            history = directory / "generations.jsonl" if directory is not None else None
            if history is not None and history.is_file():
                with history.open(encoding="utf-8") as file:
                    for line in deque(file, maxlen=30):
                        try:
                            self.events.put(dict(stats=json.loads(line)))
                        except json.JSONDecodeError:
                            break

    def mount(self, root):
        self.root = root
        outer = tk.Frame(root, bg=SURFACE, width=326)
        outer.pack(side="left", fill="y", padx=(0, 14))
        outer.pack_propagate(False)
        scroll = tk.Canvas(outer, bg=SURFACE, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=scroll.yview)
        scrollbar.pack(side="right", fill="y")
        scroll.pack(side="left", fill="both", expand=True)
        scroll.configure(yscrollcommand=scrollbar.set)
        inner = tk.Frame(scroll, bg=SURFACE, padx=12, pady=8)
        window = scroll.create_window(0, 0, window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _event: scroll.configure(scrollregion=scroll.bbox("all")))
        scroll.bind("<Configure>", lambda event: scroll.itemconfigure(window, width=event.width))
        def scroll_sidebar(event):
            left = outer.winfo_rootx()
            if event.widget is not self.log and left <= event.x_root < left + outer.winfo_width():
                scroll.yview_scroll(int(-event.delta / 120), "units")

        root.bind("<MouseWheel>", scroll_sidebar)

        header = tk.Frame(inner, bg=SURFACE)
        header.pack(fill="x", pady=(0, 4))
        self._label(header, "Neuroévolution", size=17, bold=True).pack(side="left")
        if self.settings is not None:
            self.pause_button = ttk.Button(header, text="Pause", command=self._toggle_pause, style="Panel.TButton")
            self.pause_button.pack(side="right")
        self.status = self._label(inner, "Démarrage", color=ACCENT, size=10, bold=True)
        self.status.pack(anchor="w", pady=(0, 6))
        self.training_text = self._label(inner, color=MUTED)
        self.training_text.pack(anchor="w")

        champion = self._section(inner, "Meilleur historique")
        self.record_score = self._label(champion, "En attente", color=YELLOW, size=17, bold=True)
        self.record_score.pack(anchor="w")
        self.record_text = self._label(champion, color=MUTED)
        self.record_text.pack(anchor="w")

        statistics = self._section(inner, "Dernière génération complète")
        self.stats_text = self._label(statistics, "En attente", color=MUTED)
        self.stats_text.pack(anchor="w")
        self.chart = tk.Canvas(statistics, bg=SURFACE, height=52, highlightthickness=0, borderwidth=0)
        self.chart.pack(fill="x", pady=(3, 0))
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())

        network = self._section(inner, "Réseau", command=self._enlarge)
        self.graph = tk.Canvas(network, bg=BACKGROUND, height=244, highlightthickness=0, borderwidth=0)
        self.graph.pack(fill="x")
        self.graph.bind("<Configure>", lambda _event: _draw_network(self.graph, self.genome))
        legend = tk.Frame(network, bg=SURFACE)
        legend.pack(fill="x")
        self._label(legend, "Poids +", color=ACCENT, size=8).pack(side="left", padx=(0, 8))
        self._label(legend, "Poids −", color=NEGATIVE, size=8).pack(side="left")

        events = self._section(inner, "Événements")
        log_scroll = ttk.Scrollbar(events, orient="vertical")
        log_scroll.pack(side="right", fill="y")
        self.log = tk.Text(events, bg=BACKGROUND, fg=FOREGROUND, font=(FONT, 9), wrap="word",
                           height=6, highlightthickness=0, borderwidth=0, padx=6, pady=3,
                           state="disabled", yscrollcommand=log_scroll.set)
        self.log.pack(fill="both", expand=True)
        log_scroll.configure(command=self.log.yview)
        if self.settings is not None:
            root.protocol("WM_DELETE_WINDOW", self._close)

    def _label(self, parent, text="", *, size=9, color=FOREGROUND, bold=False):
        return tk.Label(parent, text=text, bg=SURFACE, fg=color, justify="left", anchor="w", wraplength=276,
                        font=(FONT, size, "bold" if bold else "normal"))

    def _section(self, parent, title, command=None):
        section = tk.Frame(parent, bg=SURFACE)
        section.pack(fill="x", pady=(10, 0))
        tk.Frame(section, height=1, bg=BORDER_COLOR).pack(fill="x", pady=(0, 5))
        header = tk.Frame(section, bg=SURFACE)
        header.pack(fill="x", pady=(0, 3))
        self._label(header, title, bold=True, size=10).pack(side="left")
        if command is not None:
            ttk.Button(header, text="Agrandir", style="Panel.TButton", command=command).pack(side="right")
        return section

    def _enlarge(self):
        if self.zoom is not None and self.zoom.winfo_exists():
            self.zoom.winfo_toplevel().lift()
            return
        window = tk.Toplevel(self.root)
        window.title("CDFR 2027 · Réseau du champion")
        window.geometry("900x640")
        window.minsize(480, 370)
        self.zoom = tk.Canvas(window, bg=BACKGROUND, highlightthickness=0)
        self.zoom.pack(fill="both", expand=True)
        self.zoom.bind("<Configure>", lambda _event: _draw_network(self.zoom, self.genome))

    def _draw_chart(self):
        canvas = self.chart
        canvas.delete("all")
        stats = self.stats[-30:]
        if len(stats) < 2:
            return
        width, height = canvas.winfo_width(), canvas.winfo_height()
        low = min(stat["mean"] for stat in stats)
        span = max(1, max(stat["best"] for stat in stats) - low)
        for field, color in (("mean", ACCENT), ("best", YELLOW)):
            coords = [coord for i, stat in enumerate(stats)
                      for coord in (5 + (width - 10) * i / (len(stats) - 1),
                                    height - 7 - (height - 14) * (stat[field] - low) / span)]
            canvas.create_line(*coords, fill=color, width=2)
        canvas.create_text(6, 2, anchor="nw", text=f"Générations {stats[0]['generation']}–{stats[-1]['generation']}",
                           font=(FONT, 7), fill=MUTED)

    def _display_progress(self):
        progress = self.progress
        status = progress.get("status", "Démarrage")
        self.status.configure(text=status, fg=WARNING if status in {"Erreur", "Pause", "Pause demandée"} else ACCENT)
        if self.settings is None:
            self.training_text.configure(text="Table officielle · match de 100 s")
        else:
            elapsed = progress.get("elapsed", 0) if status in {"Terminé", "Erreur"} else perf_counter() - self.started
            count = progress.get("simulations", 0)
            limit = self.settings.generations
            budget = f" / {limit}" if limit else ""
            self.training_text.configure(text=(
                f"Génération {progress.get('generation', 1)} · {progress.get('completed', 0)} terminées{budget}\n"
                f"{progress.get('evaluated', 0)} / {self.settings.population} individus évalués\n"
                f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d} · "
                f"{count} simulations · {count / max(elapsed, 0.001):.2f}/s"
            ))
            if status in {"Terminé", "Erreur"}:
                self.pause_button.state(["disabled"])

    def poll(self):
        incoming = None
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            self.progress.update(event)
            if "log" in event:
                self.log.configure(state="normal")
                self.log.insert("end", event["log"] + "\n")
                if int(self.log.index("end-1c").split(".")[0]) > 100:
                    self.log.delete("1.0", "2.0")
                self.log.see("end")
                self.log.configure(state="disabled")
            if "stats" in event:
                stats = event["stats"]
                self.stats.append(stats)
                self.stats = self.stats[-30:]
                distribution = "  ".join(f"{score}×{count}" for score, count in stats["distribution"])
                self.stats_text.configure(text=(f"{stats['best']} pts en {stats['time']:.2f} s · génération {stats['generation']}\n"
                                                f"Moyenne {stats['mean']:.1f} · médiane {stats['median']:g}\n"
                                                f"Points × effectif  {distribution}"))
                self._draw_chart()
            if "champion" in event:
                data, replay = event["champion"], event["replay"]
                self.genome = Genome.read(data["genome"])
                self.record_score.configure(text=f"{data['score']} pts en {data['time']:.2f} s")
                self.record_text.configure(text=(
                    f"Réseau #{data['id']} · génération {data['generation']}\n"
                    f"{self.genome.output + 1} neurones · {len(self.genome.links)} connexions\n"
                    f"Fitness ({data['score']}, {-data['time']:.2f})"
                ))
                _draw_network(self.graph, self.genome)
                if self.zoom is not None and self.zoom.winfo_exists():
                    _draw_network(self.zoom, self.genome)
                incoming = data["id"], replay
        self._display_progress()
        return incoming

    def _toggle_pause(self):
        if self.pause.is_set():
            self.pause.clear()
            self.pause_button.configure(text="Pause")
        else:
            self.pause.set()
            self.pause_button.configure(text="Reprendre")

    def _close(self):
        self.stop.set()
        self.root.destroy()

    def work(self):
        def publish(event):
            if "champion" in event:
                event["replay"] = Replay(event["champion"]["replay"])
            self.events.put(event)

        try:
            directory = new_directory(self.directory)
            train(self.settings, directory, publish, initial=self.initial, pause=self.pause, stop=self.stop)
        except Exception as exc:
            publish(dict(status="Erreur", elapsed=perf_counter() - self.started, log=str(exc)))


def run_window(*, settings=None, directory: Path | None = None, saved=None, initial=None):
    dashboard = Dashboard(settings, directory, saved, initial)
    if saved is None:
        simulation = Simulation()
        replay = Replay(dict(obstacles=simulation.obstacles, frames=[_frame(simulation, "En attente du premier réseau")]))
    else:
        replay = saved[1]
    root = create_window(replay=replay, dashboard=dashboard)
    if settings is None:
        root.mainloop()
        return
    worker = Thread(target=dashboard.work, name="neat")
    worker.start()
    try:
        root.mainloop()
    finally:
        dashboard.stop.set()
        worker.join()
