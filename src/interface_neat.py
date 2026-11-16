"""Tableau de bord NEAT et replay du champion dans la fenêtre du simulateur."""

from collections import deque
import json
from multiprocessing import get_context
from pathlib import Path
from queue import Empty, SimpleQueue
from time import perf_counter
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font

from .neat import INPUTS, N_INPUTS, Genome, train_background
from .physique.moteur import Simulation
from .replay.lecture import Replay
from .replay.recorder import _frame
from .ui.theme import Theme
from .ui.window import create_window


THEME = Theme()
ACCENT, BACKGROUND, BORDER_COLOR = THEME.accent, THEME.background, THEME.border_color
FONT, FOREGROUND, MUTED = THEME.font, THEME.foreground, THEME.muted
SURFACE, WARNING, YELLOW = THEME.surface, THEME.warning, THEME.yellow


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
    input_font = (FONT, 7 if compact else 10)
    row_height = Font(root=canvas, font=input_font).metrics("linespace")
    show_input_labels = height - 37 >= row_height * (len(columns[0]) + 1)
    if not show_input_labels:
        left = 12
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
            if show_input_labels:
                canvas.create_text(x - 8, y, anchor="e", text=INPUTS[node], font=input_font, fill=FOREGROUND)
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
    def __init__(self, settings, directory, saved):
        self.settings = settings
        self.directory = directory
        self.context = get_context("spawn")
        self.events = self.context.Queue() if settings is not None else SimpleQueue()
        self.stop, self.pause = (self.context.Event(), self.context.Event()) if settings is not None else (None, None)
        self.worker = None
        self.progress = {}
        self.stats = deque(maxlen=30)
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
        sidebar = tk.Frame(root, bg=SURFACE, width=258, padx=10, pady=10)
        sidebar.pack(side="left", fill="y", padx=(0, 12))
        sidebar.pack_propagate(False)
        header = tk.Frame(sidebar, bg=SURFACE)
        header.pack(fill="x", pady=(0, 4))
        self.status = self._label(header, "Démarrage", color=ACCENT, size=10, bold=True)
        self.status.pack(side="left")
        if self.settings is not None:
            self.pause_button = ttk.Button(header, text="Pause", command=self._toggle_pause, style="Panel.TButton")
            self.pause_button.pack(side="right")
        self.generation = self._label(sidebar, size=15, bold=True)
        self.generation.pack(anchor="w")
        self.training_text = self._label(sidebar, color=MUTED)
        self.training_text.pack(anchor="w")
        if self.settings is not None:
            self.stop_button = ttk.Button(sidebar, text="Arrêter et sauvegarder", command=self._request_stop,
                                          style="Panel.TButton")
            self.stop_button.pack(fill="x", pady=(6, 0))

        champion = self._section(sidebar, "Meilleur historique")
        self.record_score = self._label(champion, "En attente", color=YELLOW, size=15, bold=True)
        self.record_score.pack(anchor="w")

        statistics = self._section(sidebar, "Évolution par génération")
        self.chart = tk.Canvas(statistics, bg=BACKGROUND, height=120, highlightthickness=0)
        self.chart.pack(fill="x")
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())

        network = self._section(sidebar, "Réseau", command=self._enlarge, expand=True)
        legend = tk.Frame(network, bg=SURFACE)
        legend.pack(side="bottom", fill="x")
        self._label(legend, "Poids +", color=ACCENT, size=8).pack(side="left", padx=(0, 8))
        self._label(legend, "Poids −", color=NEGATIVE, size=8).pack(side="left")
        self.graph = tk.Canvas(network, bg=BACKGROUND, height=1, highlightthickness=0)
        self.graph.pack(fill="both", expand=True)
        self.graph.bind("<Configure>", lambda _event: _draw_network(self.graph, self.genome))

        events = tk.Frame(root, bg=SURFACE, width=194, padx=8, pady=10)
        events.pack(side="right", fill="y", padx=(12, 0))
        events.pack_propagate(False)
        self._label(events, "Logs", bold=True, size=10).pack(anchor="w", pady=(0, 6))
        log_scroll = ttk.Scrollbar(events, orient="vertical")
        log_scroll.pack(side="right", fill="y")
        self.log = tk.Text(events, bg=BACKGROUND, fg=FOREGROUND, font=(FONT, 9), wrap="word",
                           highlightthickness=0, borderwidth=0, padx=6, pady=3,
                           state="disabled", yscrollcommand=log_scroll.set)
        self.log.pack(fill="both", expand=True)
        log_scroll.configure(command=self.log.yview)
        if self.settings is not None:
            root.protocol("WM_DELETE_WINDOW", self._close)

    def _label(self, parent, text="", *, size=9, color=FOREGROUND, bold=False):
        return tk.Label(parent, text=text, bg=SURFACE, fg=color, anchor="w",
                        font=(FONT, size, "bold" if bold else "normal"))

    def _section(self, parent, title, *, command=None, expand=False):
        section = tk.Frame(parent, bg=SURFACE)
        section.pack(fill="both" if expand else "x", expand=expand, pady=(10, 0))
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
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        if not self.stats:
            canvas.create_text(width / 2, height / 2, text="En attente de résultats", font=(FONT, 9), fill=MUTED)
            return
        left, right = 7, width - 7
        row_height = (height - 17) / 2
        for row, (field, title, color) in enumerate((("best", "Score", YELLOW), ("time", "Temps du score", ACCENT))):
            values = [stat[field] for stat in self.stats]
            label = f"{values[-1]} pts" if field == "best" else f"{values[-1]:.2f} s"
            top = row * row_height
            canvas.create_text(left, top + 3, anchor="nw", text=title, font=(FONT, 8), fill=MUTED)
            canvas.create_text(right, top + 3, anchor="ne", text=label, font=(FONT, 8, "bold"), fill=color)
            low, high = min(values), max(values)
            if low == high:
                low, high = low - 0.5, high + 0.5
            bottom = top + row_height - 6
            step = (right - left) / max(1, len(values) - 1)
            scale = (row_height - 28) / (high - low)
            points = [(left + i * step, bottom - (value - low) * scale) for i, value in enumerate(values)]
            canvas.create_line(left, bottom + 3, right, bottom + 3, fill=BORDER_COLOR)
            if len(points) > 1:
                canvas.create_line(*(value for point in points for value in point), fill=color, width=2)
            x, y = points[-1]
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")
        canvas.create_text(left, height - 3, anchor="sw", text=f"Gén. {self.stats[0]['generation']}",
                           font=(FONT, 7), fill=MUTED)
        if len(self.stats) > 1:
            canvas.create_text(right, height - 3, anchor="se", text=f"Gén. {self.stats[-1]['generation']}",
                               font=(FONT, 7), fill=MUTED)

    def _display_progress(self):
        progress = self.progress
        status = progress.get("status", "Démarrage")
        finished = status in {"Terminé", "Erreur"}
        stopping = self.stop is not None and self.stop.is_set()
        if stopping and not finished:
            status = "Arrêt en cours"
        self.status.configure(text=status, fg=WARNING if status in {"Erreur", "Pause", "Pause demandée", "Arrêt en cours"} else ACCENT)
        if self.settings is None:
            progress = self.stats[-1] if self.stats else progress.get("champion", {})
            generation = progress.get("generation")
            elapsed = progress.get("elapsed")
        else:
            generation = progress.get("generation", 1)
            elapsed = progress.get("elapsed", 0) if finished else perf_counter() - self.started
            if finished or stopping:
                self.pause_button.state(["disabled"])
                self.stop_button.state(["disabled"])
        self.generation.configure(text=f"Génération {generation}" if generation is not None else "")
        if elapsed is not None:
            minutes, seconds = divmod(int(elapsed), 60)
            count = progress.get("simulations", 0)
            self.training_text.configure(text=f"{minutes:02d}:{seconds:02d} · {count} sim. · {count / max(elapsed, 0.001):.2f}/s")

    def _append_log(self, message):
        at_end = self.log.yview()[1] >= 1
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 100:
            self.log.delete("1.0", "2.0")
        if at_end:
            self.log.see("end")
        self.log.configure(state="disabled")

    def poll(self):
        incoming = None
        chart_changed = False
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            self.progress.update(event)
            if "log" in event:
                self._append_log(event["log"])
            if "stats" in event:
                self.stats.append(event["stats"])
                chart_changed = True
            if "champion" in event:
                data, replay = event["champion"], event["replay"]
                self.genome = Genome.read(data["genome"])
                self.record_score.configure(text=f"{data['score']} pts en {data['time']:.2f} s")
                _draw_network(self.graph, self.genome)
                if self.zoom is not None and self.zoom.winfo_exists():
                    _draw_network(self.zoom, self.genome)
                incoming = data["id"], replay
        if chart_changed:
            self._draw_chart()
        if (self.worker is not None and self.worker.exitcode not in (None, 0)
                and self.progress.get("status") != "Erreur"):
            self.progress.update(status="Erreur", elapsed=perf_counter() - self.started)
            self._append_log(f"Le processus d’entraînement s’est arrêté · code {self.worker.exitcode}")
        self._display_progress()
        return incoming

    def _toggle_pause(self):
        if self.pause.is_set():
            self.pause.clear()
            self.pause_button.configure(text="Pause")
        else:
            self.pause.set()
            self.pause_button.configure(text="Reprendre")

    def _request_stop(self):
        self.stop.set()
        self._display_progress()

    def _close(self):
        self.stop.set()
        self.root.destroy()

    def start(self, initial=None):
        self.worker = self.context.Process(target=train_background, name="neat", args=(
            self.settings, self.directory, initial, self.events, self.pause, self.stop,
        ))
        self.worker.start()

    def join(self):
        self.stop.set()
        # Vider la file tant que le processus peut publier un dernier champion.
        while self.worker.is_alive():
            try:
                self.events.get(timeout=0.2)
            except Empty:
                pass
        self.worker.join()
        self.events.close()


def run_window(*, settings=None, directory: Path | None = None, saved=None, initial=None):
    dashboard = Dashboard(settings, directory, saved)
    if saved is None:
        simulation = Simulation()
        replay = Replay(dict(obstacles=simulation.obstacles, frames=[_frame(simulation, "En attente du premier réseau")]))
    else:
        replay = saved[1]
    root = create_window(replay=replay, dashboard=dashboard)
    if settings is None:
        root.mainloop()
        return
    dashboard.start(initial)
    try:
        root.mainloop()
    finally:
        dashboard.join()
