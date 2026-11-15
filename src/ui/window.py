from math import cos, degrees, sin
from time import perf_counter

from ..simulateur import TABLE_HEIGHT, TABLE_WIDTH, TURN_STEP, Simulation
from .assets import charger_tapis
from .canvas_renderer import etat_robot
from .manual_controller import ControleurManuel, RepeteurClavier
from .replay_controller import ControleurReplayTk
from .theme import Theme

BORDER = 22


def create_window(simulation: Simulation | None = None, *, replay=None, trial=None):
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    theme = Theme()
    background = theme.background
    surface = theme.surface
    border_color = theme.border_color
    foreground = theme.foreground
    muted = theme.muted
    accent = theme.accent
    yellow = theme.yellow
    warning = theme.warning
    font = theme.font
    root = tk.Tk()
    mode = "Pilotage manuel" if replay is None else f"Replay · Essai {trial}"
    root.title(f"CDFR 2027 · {mode}")
    root.configure(bg=background, padx=20, pady=16)
    width = min(1000, root.winfo_screenwidth() - 48)
    height = min(780, root.winfo_screenheight() - 88)
    root.geometry(f"{width}x{height}")
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda _event: root.attributes("-fullscreen", False))
    root.bind("<F11>", lambda _event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
    content = tk.Frame(root, bg=background)
    content.pack(fill="both", expand=True)
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

    header = tk.Frame(content, bg=background)
    header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    heading = tk.Frame(header, bg=background)
    heading.pack(side="left")
    tk.Label(
        heading, text="CDFR 2027", font=(font, 20, "bold"),
        bg=background, fg=foreground,
    ).pack(anchor="w")
    tk.Label(heading, text=mode, bg=background, fg=muted, font=(font, 10)).pack(anchor="w")

    scoreboard = tk.Frame(header, bg=surface, padx=12, pady=5)
    scoreboard.pack(side="right")
    tk.Label(scoreboard, text="Score jaune", bg=surface, fg=muted, font=(font, 9)).pack(side="left")
    score = tk.Label(scoreboard, bg=surface, fg=yellow, font=(font, 19, "bold"), width=3)
    score.pack(side="left", padx=(8, 0))

    details = tk.Frame(header, bg=background)
    details.pack(side="left", fill="x", expand=True, padx=12)
    status = tk.Label(details, bg=background, fg=foreground, font=(font, 10))
    status.pack(anchor="e")
    readout = tk.Label(details, bg=background, fg=muted, font=(font, 9))
    readout.pack(anchor="e")

    mat = charger_tapis()
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
        for _ in (simulation.blocks if replay is None else replay.frame["piles"])
    ]
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
            target, blocked, label = etat_robot(simulation)
        else:
            frame = replay.frame
            points, position, angle = frame["score"], frame["position"], frame["heading"]
            footprint, piles, cups = frame["robot"], frame["piles"], frame["cups"]
            target, blocked, label = frame["target"], frame["blocked"], frame["action"]
            readout.configure(text=f"{replay.time:.1f} s · fin à {replay.duration:.1f} s")

        score.configure(text=str(points))
        for item, corners in zip(block_items, piles):
            canvas.coords(item, *(border + value * scale for corner in corners for value in corner))
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
        ControleurReplayTk(root, footer, replay, surface, draw, tk=tk, ttk=ttk)
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

    manuel = ControleurManuel(canvas, simulation, cup_targets=cup_targets,
                              etat=lambda: (scale, border))
    click = manuel.click
    nudge = manuel.nudge

    clavier = RepeteurClavier(root)
    press = clavier.press
    release = clavier.release
    release_all = clavier.release_all

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
    for key, dx, dy in (("Left", -100, 0), ("Right", 100, 0), ("Up", 0, -100), ("Down", 0, 100)):
        root.bind(f"<KeyPress-{key}>", lambda event, dx=dx, dy=dy: press(event, lambda: nudge(dx, dy)))
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
        root.after(16, tick)

    tick()
    return root


if __name__ == "__main__":
    try:
        simulation = Simulation()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    create_window(simulation).mainloop()
