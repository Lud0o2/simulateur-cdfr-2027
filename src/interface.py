from bisect import bisect_right
from math import cos, degrees, hypot, sin
from pathlib import Path
from time import perf_counter

from .simulateur import TABLE_HEIGHT, TABLE_WIDTH, TURN_STEP, Simulation

BORDER = 22
BACKGROUND = "#111820"
SURFACE = "#1b2732"
BORDER_COLOR = "#344450"
FOREGROUND = "#f0f4f6"
MUTED = "#aabac5"
ACCENT = "#84e0cb"
YELLOW = "#f5d477"
WARNING = "#ffbc75"
FONT = "Segoe UI"


class Playback:
    def __init__(self, replay, trial, *, loop=False):
        self.replay = replay
        self.trial = trial
        self.loop = loop
        self.playing = True
        self.speed = 1.0
        self.pending = None

    def offer(self, trial, replay):
        self.pending = trial, replay
        if self.trial is None:
            self.restart()

    def restart(self):
        if self.pending is not None:
            self.trial, self.replay = self.pending
            self.pending = None
        self.replay.time = self.replay.start

    def advance(self, seconds):
        if not self.playing or (self.trial is None and self.replay.start == self.replay.duration):
            return False
        if self.loop and self.replay.time >= self.replay.duration:
            self.restart()
            return True
        self.replay.time = min(self.replay.duration, self.replay.time + seconds * self.speed)
        if not self.loop and self.replay.time >= self.replay.duration:
            self.playing = False
        return True


def create_window(simulation: Simulation | None = None, *, replay=None, trial=None, dashboard=None):
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    background, surface, border_color, foreground = BACKGROUND, SURFACE, BORDER_COLOR, FOREGROUND
    muted, accent, yellow, warning, font = MUTED, ACCENT, YELLOW, WARNING, FONT
    root = tk.Tk()
    mode = "Pilotage manuel" if replay is None else f"Replay · Essai {trial}"
    if dashboard is not None:
        mode = "Replay du meilleur réseau"
    root.title(f"CDFR 2027 · {mode}")
    root.configure(bg=background, padx=20, pady=16)
    width = min(1000, root.winfo_screenwidth() - 48)
    height = min(780, root.winfo_screenheight() - 88)
    root.geometry(f"{width}x{height}")
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda _event: root.attributes("-fullscreen", False))
    root.bind("<F11>", lambda _event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
    updates = {}

    def schedule(delay, update):
        updates[update] = root.after(delay, update)

    def cancel_updates(event):
        if event.widget is root:
            for timer in updates.values():
                root.after_cancel(timer)

    root.bind("<Destroy>", cancel_updates)
    content = tk.Frame(root, bg=background)
    content.columnconfigure(0, weight=1)
    content.rowconfigure(1, weight=1)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "Panel.TButton", background=surface, foreground=foreground, font=(font, 10, "bold"),
        bordercolor=border_color, lightcolor=surface, darkcolor=surface, focuscolor=accent,
        padding=(12, 5), relief="flat",
    )
    style.map("Panel.TButton", background=[("pressed", border_color), ("active", "#2a3a47")])
    style.configure(
        "Replay.Horizontal.TScale", background=accent, troughcolor=border_color,
        bordercolor=surface, lightcolor=accent, darkcolor=accent, focuscolor=accent,
        sliderlength=14, arrowsize=9, gripcount=0,
    )
    if dashboard is not None:
        dashboard.mount(root)
    content.pack(fill="both", expand=True)
    playback = Playback(replay, trial, loop=dashboard is not None) if replay is not None else None

    header = tk.Frame(content, bg=background)
    header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    heading = tk.Frame(header, bg=background)
    heading.pack(side="left")
    tk.Label(
        heading, text="CDFR 2027", font=(font, 20, "bold"),
        bg=background, fg=foreground,
    ).pack(anchor="w")
    subtitle = tk.Label(heading, text=mode, bg=background, fg=muted, font=(font, 10))
    subtitle.pack(anchor="w")

    scoreboard = tk.Frame(header, bg=surface, padx=12, pady=5)
    scoreboard.pack(side="right")
    collection = replay is not None and replay.collection
    score_label = "Piles ramenées" if collection else "Score jaune"
    tk.Label(scoreboard, text=score_label, bg=surface, fg=muted, font=(font, 9)).pack(side="left")
    score = tk.Label(scoreboard, bg=surface, fg=yellow, font=(font, 19, "bold"), width=3)
    score.pack(side="left", padx=(8, 0))

    details = tk.Frame(header, bg=background)
    details.pack(side="left", fill="x", expand=True, padx=12)
    status = tk.Label(details, bg=background, fg=foreground, font=(font, 10))
    status.pack(anchor="e")
    readout = tk.Label(details, bg=background, fg=muted, font=(font, 9))
    readout.pack(anchor="e")

    with Image.open(Path(__file__).resolve().parents[1] / "assets" / "tapis_2027.png") as image:
        mat = image.copy()
    scale = border = 0
    viewport = tk.Frame(content, bg=background)
    viewport.grid(row=1, column=0, sticky="nsew")
    board = tk.Frame(viewport, bg=border_color, padx=1, pady=1)
    board.place(relx=0.5, rely=0.5, anchor="center")
    canvas = tk.Canvas(
        board, width=1, height=1, bg="#b5b0a1", highlightthickness=0,
    )
    canvas.pack()
    mat_item = canvas.create_image(0, 0, anchor="nw")
    obstacles = simulation.obstacles if replay is None else replay.obstacles
    obstacle_items = [
        canvas.create_rectangle(0, 0, 0, 0, fill="#474e56", outline=foreground)
        for _ in obstacles
    ]
    block_items = [
        canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0, fill="#c99d68", outline="#553b24", width=2)
        for _ in (simulation.blocks if replay is None else replay.frames[0]["piles"])
    ]
    trace = canvas.create_line(0, 0, 0, 0, fill="#8baea1", width=2, state="hidden") if dashboard else None
    path = canvas.create_line(0, 0, 0, 0, fill=accent, width=2, dash=(4, 5), state="hidden")
    cups_at_start = simulation.suction_cups if replay is None else replay.frame["cups"]
    # Le châssis masque les supports intérieurs, y compris dans les anciens replays.
    supports = [canvas.create_line(0, 0, 0, 0, fill=accent, width=2) for _ in cups_at_start]
    robot = canvas.create_polygon(0, 0, 0, 0, 0, 0, fill=background, outline=accent, width=2)
    front = canvas.create_line(0, 0, 0, 0, fill=foreground, width=2, arrow="last", arrowshape=(7, 8, 3))
    cup_items = [
        (canvas.create_oval(0, 0, 0, 0, outline=foreground),
         canvas.create_text(0, 0, text=str(number), font=(font, 8, "bold")))
        for number, _ in enumerate(cups_at_start, 1)
    ]
    panel = surface if replay is not None else background
    footer = tk.Frame(content, bg=panel, padx=10 if replay is not None else 0, pady=6)
    footer.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def draw():
        if replay is None:
            points, position, angle = simulation.score, simulation.position, simulation.heading
            footprint = simulation.footprint
            piles = (block.corners for block in simulation.blocks)
            cups = ((cup.position, cup.enabled) for cup in simulation.suction_cups)
            move = simulation.movement
            moving = simulation.moving
            target = move.end if moving and not move.turn else None
            blocked = move is not None and (move.blocked or move.rotation_blocked)
            if target is not None:
                label = "En déplacement"
            elif moving:
                label = "En rotation"
            elif move is None:
                label = "Au départ"
            elif move.rotation_blocked:
                label = "Rotation bloquée"
            elif move.turn:
                label = "Rotation bloquée" if blocked else "Cap réglé"
            else:
                label = "Trajet bloqué" if blocked else "Arrivé"
        else:
            current = playback.replay
            frame = current.frame
            points, position, angle = frame["score"], frame["position"], frame["heading"]
            footprint, piles, cups = frame["robot"], frame["piles"], frame["cups"]
            target, blocked, label = frame["target"], frame["blocked"], frame["action"]
            time = f"{current.time:.1f} s · fin à {current.duration:.1f} s"
            if dashboard is not None:
                subtitle.configure(text=f"Replay · Réseau #{playback.trial}" if playback.trial is not None else mode)
                x, y = position
                time += f"  ·  X {x:.0f}  Y {y:.0f} mm  ·  {degrees(angle) % 360:.0f}°"
                start = bisect_right(current.times, current.time - 5)
                end = bisect_right(current.times, current.time)
                trail = [border + value * scale for previous in current.frames[start:end:2]
                         for value in previous["position"]]
                if len(trail) >= 4:
                    canvas.coords(trace, *trail)
                canvas.itemconfigure(trace, state="normal" if len(trail) >= 4 else "hidden")
            readout.configure(text=time)

        score.configure(text=str(points))
        for item, corners in zip(block_items, piles):
            canvas.coords(item, *(border + value * scale for corner in corners for value in corner))
            if collection:
                canvas.itemconfigure(item, state="normal")
        if collection:
            for item in block_items[len(piles):]:
                canvas.itemconfigure(item, state="hidden")
        x, y = position
        screen_x, screen_y = border + scale * x, border + scale * y
        vertices = [(screen_x + scale * dx, screen_y + scale * dy) for dx, dy in footprint]
        canvas.coords(robot, *(coordinate for vertex in vertices for coordinate in vertex))
        nose = 65 * scale
        canvas.coords(front, screen_x, screen_y, screen_x + nose * cos(angle), screen_y + nose * sin(angle))
        center_x, center_y = (sum(axis) / len(vertices) for axis in zip(*vertices))
        for support, (oval, number), (mount, enabled) in zip(supports, cup_items, cups):
            cx, cy = (border + scale * value for value in mount)
            canvas.coords(support, center_x, center_y, cx, cy)
            canvas.coords(oval, cx - 8, cy - 8, cx + 8, cy + 8)
            canvas.coords(number, cx, cy)
            canvas.itemconfigure(oval, fill=warning if enabled else "#474e56")
            canvas.itemconfigure(number, fill=background if enabled else foreground)
        color = warning if blocked else accent
        if target is not None:
            end_x, end_y = target
            canvas.coords(path, screen_x, screen_y, border + scale * end_x, border + scale * end_y)
            canvas.itemconfigure(path, state="normal", fill=color)
        else:
            canvas.itemconfigure(path, state="hidden")
        canvas.itemconfigure(robot, outline=color)
        if replay is None:
            readout.configure(text=f"X {x:.0f}  Y {y:.0f} mm  ·  {degrees(angle) % 360:.0f}°")
        status.configure(text=label, fg=warning if blocked else foreground)

    def fit_board(event):
        nonlocal scale, border
        fit = min((event.width - 2) / (TABLE_WIDTH + 2 * BORDER),
                  (event.height - 2) / (TABLE_HEIGHT + 2 * BORDER))
        # Des dimensions multiples de 3 et de 2 gardent le rapport 3:2 au pixel près.
        fitted_scale = int(fit * 1000) / 1000
        if fitted_scale <= 0 or fitted_scale == scale:
            return
        scale = fitted_scale
        border = int(BORDER * scale)
        size = round(TABLE_WIDTH * scale), round(TABLE_HEIGHT * scale)
        canvas.mat = ImageTk.PhotoImage(mat.resize(size, Image.Resampling.LANCZOS), master=root)
        canvas.configure(width=size[0] + 2 * border, height=size[1] + 2 * border)
        canvas.coords(mat_item, border, border)
        canvas.itemconfigure(mat_item, image=canvas.mat)
        for item, rectangle in zip(obstacle_items, obstacles):
            canvas.coords(item, *(border + value * scale for value in rectangle))
        draw()

    viewport.bind("<Configure>", fit_board)

    if replay is not None:
        dragging = False
        last_frame = perf_counter()

        def refresh():
            current = playback.replay
            finished = current.time >= current.duration
            if finished and not playback.loop and not dragging:
                playback.playing = False
            button.configure(text="Pause" if playback.playing else "Relancer" if finished else "Reprendre")
            draw()

        def sync_timeline():
            current = playback.replay
            timeline.configure(from_=current.start, to=current.duration)
            timeline.set(current.time)
            refresh()

        def restart():
            nonlocal last_frame
            playback.restart()
            playback.playing = True
            last_frame = perf_counter()
            sync_timeline()

        def toggle():
            nonlocal last_frame
            playback.playing = not playback.playing
            last_frame = perf_counter()
            if playback.playing and playback.replay.time >= playback.replay.duration:
                restart()
            else:
                refresh()

        def seek(value):
            playback.replay.time = float(value)
            refresh()

        def drag(event):
            nonlocal dragging
            dragging = True
            timeline.focus_set()
            timeline.set(timeline.get(event.x, event.y))
            return "break"

        def release_drag(_event):
            nonlocal dragging, last_frame
            dragging = False
            last_frame = perf_counter()
            refresh()

        def tick_replay():
            nonlocal last_frame
            now = perf_counter()
            if not dragging and playback.advance(now - last_frame):
                if playback.replay.duration != float(timeline.cget("to")):
                    sync_timeline()
                else:
                    timeline.set(playback.replay.time)
            last_frame = now
            schedule(16, tick_replay)

        def poll_training():
            incoming = dashboard.poll()
            if incoming is not None:
                playback.offer(*incoming)
                sync_timeline()
            schedule(150, poll_training)

        controls = tk.Frame(footer, bg=surface)
        controls.pack(fill="x")
        button = ttk.Button(controls, command=toggle, width=10, style="Panel.TButton")
        button.pack(side="left", padx=(0, 10))
        if dashboard is not None:
            ttk.Button(controls, text="Recommencer", command=restart, style="Panel.TButton").pack(
                side="left", padx=(0, 10),
            )
        timeline = ttk.Scale(
            controls, from_=replay.start, to=replay.duration, value=replay.time, command=seek,
            style="Replay.Horizontal.TScale", cursor="hand2",
        )
        timeline.pack(side="left", expand=True, fill="x")
        timeline.bind("<Button-1>", drag)
        timeline.bind("<B1-Motion>", drag)
        timeline.bind("<ButtonRelease-1>", release_drag)
        if dashboard is not None:
            loop = tk.BooleanVar(master=root, value=True)
            speed = tk.StringVar(master=root, value="1")
            style.configure("Replay.TCheckbutton", background=surface, foreground=foreground, font=(font, 9))
            ttk.Checkbutton(controls, text="Boucle", variable=loop, style="Replay.TCheckbutton",
                            command=lambda: setattr(playback, "loop", loop.get())).pack(side="left", padx=8)
            tk.Label(controls, text="×", bg=surface, fg=foreground, font=(font, 10)).pack(side="left")
            speeds = ttk.Combobox(controls, textvariable=speed, state="readonly", width=4,
                                  values=("0,5", "1", "2", "5", "10"))
            speeds.pack(side="left", padx=(2, 0))
            speeds.bind("<<ComboboxSelected>>", lambda _event: setattr(playback, "speed", float(speed.get().replace(",", "."))))
            poll_training()
        refresh()
        tick_replay()
        return root

    cup_targets = {item: number for number, items in enumerate(cup_items, 1) for item in items}
    digit_keys = tuple(zip("1234567890", (
        "ampersand", "eacute", "quotedbl", "apostrophe", "parenleft",
        "minus", "egrave", "underscore", "ccedilla", "agrave",
    )))[:len(cup_items)]
    help_text = "Clic / flèches  ·  Q / D"
    if digit_keys:
        key_range = f"1 à {digit_keys[-1][0]}" if 1 < len(digit_keys) < 10 else "1"
        if len(digit_keys) == 10:
            key_range = "1 à 9, 0"
        help_text += f"  ·  Ventouses {key_range}"
    tk.Label(
        footer, text=help_text, bg=background, fg=muted,
        font=(font, 9),
    ).pack(anchor="w")

    def click(event):
        if not scale:
            return
        current = canvas.find_withtag("current")
        if current and current[0] in cup_targets:
            simulation.toggle_suction(cup_targets[current[0]])
            return
        x, y = (event.x - border) / scale, (event.y - border) / scale
        if 0 <= x <= TABLE_WIDTH and 0 <= y <= TABLE_HEIGHT:
            simulation.move_to(x, y)

    held_keys: dict[int, str | None] = {}
    directions: dict[int, tuple[int, int]] = {}
    direction_timer = None

    def cancel_direction_repeat():
        nonlocal direction_timer
        if direction_timer is not None:
            root.after_cancel(direction_timer)
            direction_timer = None

    def nudge(delay=60):
        nonlocal direction_timer
        dx = sum(dx for dx, _ in directions.values())
        dy = sum(dy for _, dy in directions.values())
        distance = hypot(dx, dy)
        x, y = simulation.position
        if distance:
            simulation.move_to(x + 100 * dx / distance, y + 100 * dy / distance)
            direction_timer = root.after(delay, nudge)
        else:
            simulation.move_to(x, y)

    def press_direction(event, dx, dy):
        if event.keycode in directions:
            return
        directions[event.keycode] = (dx, dy)
        cancel_direction_repeat()
        nudge(120 if len(directions) == 1 else 60)

    def press(event, action, *, repeats=True):
        keycode = event.keycode
        if keycode in held_keys:
            return

        def repeat():
            action()
            held_keys[keycode] = root.after(60, repeat)

        action()
        held_keys[keycode] = root.after(120, repeat) if repeats else None

    def release(event):
        if directions.pop(event.keycode, None) is not None:
            cancel_direction_repeat()
            if directions:
                nudge()
            return
        timer = held_keys.pop(event.keycode, None)
        if timer is not None:
            root.after_cancel(timer)

    def release_all(event=None):
        cancel_direction_repeat()
        directions.clear()
        for timer in held_keys.values():
            if timer is not None:
                root.after_cancel(timer)
        held_keys.clear()

    def reset_match():
        nonlocal last_frame
        release_all()
        simulation.reset()
        last_frame = perf_counter()
        draw()

    ttk.Button(header, text="Reset match", command=reset_match, style="Panel.TButton").pack(
        side="right", before=scoreboard, padx=(0, 10),
    )
    canvas.bind("<Button-1>", click)
    root.bind("<KeyRelease>", release)
    root.bind("<FocusOut>", release_all)
    for key, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
        root.bind(f"<KeyPress-{key}>", lambda event, dx=dx, dy=dy: press_direction(event, dx, dy))
    for key, angle in (("q", -TURN_STEP), ("Q", -TURN_STEP), ("d", TURN_STEP), ("D", TURN_STEP)):
        root.bind(f"<KeyPress-{key}>", lambda event, angle=angle: press(event, lambda: simulation.turn_by(angle)))
    for number, (digit, azerty) in enumerate(digit_keys, 1):
        for key in (digit, azerty, f"KP_{digit}"):
            root.bind(f"<KeyPress-{key}>", lambda event, n=number: press(
                event, lambda: simulation.toggle_suction(n), repeats=False,
            ))

    last_frame = perf_counter()

    def tick():
        nonlocal last_frame
        now = perf_counter()
        simulation.advance(now - last_frame)
        last_frame = now
        draw()
        schedule(16, tick)

    tick()
    return root


if __name__ == "__main__":
    try:
        simulation = Simulation()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    create_window(simulation).mainloop()
