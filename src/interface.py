from math import cos, degrees, sin
from pathlib import Path
from time import perf_counter

from .simulateur import TABLE_HEIGHT, TABLE_WIDTH, TURN_STEP, Simulation

BORDER = 22


def create_window(simulation: Simulation):
    import tkinter as tk

    background = "#151d24"
    foreground = "#e8eef3"
    muted = "#b5c4cf"
    accent = "#9cf2e1"
    warning = "#ffcb84"
    root = tk.Tk()
    root.title("Simulateur CDFR 2027")
    root.configure(bg=background, padx=16, pady=12)
    root.resizable(False, False)

    header = tk.Frame(root, bg=background)
    header.pack(fill="x", pady=(0, 10))
    tk.Label(
        header, text="Simulateur CDFR 2027", font=("Segoe UI", 15, "bold"),
        bg=background, fg=foreground,
    ).pack(side="left")
    reset_button = tk.Button(header, text="Reset match")
    reset_button.pack(side="right")
    score = tk.Label(header, bg=background, fg=foreground, font=("Segoe UI", 11, "bold"))
    score.pack(side="right", padx=12)

    mat = tk.PhotoImage(file=Path(__file__).resolve().parents[1] / "assets" / "tapis_2027.png")
    if root.winfo_screenwidth() < 960 or root.winfo_screenheight() < 760:
        mat = mat.subsample(2)
    scale = mat.width() / TABLE_WIDTH
    border = BORDER * scale
    canvas = tk.Canvas(
        root, width=mat.width() + 2 * border, height=mat.height() + 2 * border,
        bg="#b5b0a1", highlightthickness=0,
    )
    canvas.pack()
    canvas.mat = mat
    canvas.create_image(border, border, image=mat, anchor="nw")
    for left, top, right, bottom in simulation.obstacles:
        canvas.create_rectangle(
            border + left * scale, border + top * scale,
            border + right * scale, border + bottom * scale,
            fill="#474e56", outline=foreground,
        )
    block_items = [
        canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0, fill="#c99d68", outline="#553b24", width=2)
        for _ in simulation.blocks
    ]
    path = canvas.create_line(0, 0, 0, 0, fill=accent, width=2, dash=(5, 4), state="hidden")
    robot = canvas.create_polygon(0, 0, 0, 0, 0, 0, fill="#152d3a", outline=accent, width=3)
    front = canvas.create_line(0, 0, 0, 0, fill=accent, width=2, arrow="last")
    cup_items = [
        (canvas.create_oval(0, 0, 0, 0, outline=foreground),
         canvas.create_text(0, 0, text=str(number), font=("Segoe UI", 9, "bold")))
        for number, _ in enumerate(simulation.suction_cups, 1)
    ]
    cup_targets = {item: number for number, items in enumerate(cup_items, 1) for item in items}
    digit_keys = tuple(zip("1234567890", (
        "ampersand", "eacute", "quotedbl", "apostrophe", "parenleft",
        "minus", "egrave", "underscore", "ccedilla", "agrave",
    )))[:len(cup_items)]
    help_text = "Clic/flèches · Q/D pivoter"
    if digit_keys:
        key_range = f"1 à {digit_keys[-1][0]}" if 1 < len(digit_keys) < 10 else "1"
        if len(digit_keys) == 10:
            key_range = "1 à 9, 0"
        help_text += f" · Ventouses {key_range} ou clic"

    footer = tk.Frame(root, bg=background)
    footer.pack(fill="x", pady=(8, 0))
    status = tk.Label(footer, bg=background, fg=foreground, font=("Segoe UI", 10, "bold"))
    status.pack(side="left")
    tk.Label(
        footer, text=help_text, bg=background, fg=muted,
        font=("Segoe UI", 10),
    ).pack(side="right")

    def draw():
        score.configure(text=f"Score jaune : {simulation.score}")
        for item, block in zip(block_items, simulation.blocks):
            canvas.coords(item, *(border + value * scale for corner in block.corners for value in corner))
        x, y = simulation.position
        screen_x, screen_y = border + scale * x, border + scale * y
        vertices = [(screen_x + scale * dx, screen_y + scale * dy) for dx, dy in simulation.footprint]
        canvas.coords(robot, *(coordinate for vertex in vertices for coordinate in vertex))
        angle = simulation.heading
        canvas.coords(front, screen_x, screen_y, screen_x + 18 * cos(angle), screen_y + 18 * sin(angle))
        for (oval, number), cup in zip(cup_items, simulation.suction_cups):
            cx, cy = (border + scale * value for value in cup.position)
            canvas.coords(oval, cx - 8, cy - 8, cx + 8, cy + 8)
            canvas.coords(number, cx, cy)
            canvas.itemconfigure(oval, fill=warning if cup.enabled else "#474e56")
            canvas.itemconfigure(number, fill=background if cup.enabled else foreground)
        move = simulation.movement
        color = warning if move is not None and move.blocked else accent
        if simulation.moving and not move.turn:
            end_x, end_y = move.end
            canvas.coords(path, screen_x, screen_y, border + scale * end_x, border + scale * end_y)
            canvas.itemconfigure(path, state="normal", fill=color)
            label = "En déplacement"
        else:
            canvas.itemconfigure(path, state="hidden")
            if simulation.moving:
                label = "En rotation"
            elif move is None:
                label = "Au départ"
            elif move.turn:
                label = "Rotation bloquée" if move.blocked else "Cap réglé"
            else:
                label = "Trajet bloqué" if move.blocked else "Arrivé"
        canvas.itemconfigure(robot, outline=color)
        status.configure(text=f"{label}  ·  X {x:.0f}  Y {y:.0f} mm  ·  {degrees(simulation.heading) % 360:.0f}°")

    def click(event):
        current = canvas.find_withtag("current")
        if current and current[0] in cup_targets:
            simulation.toggle_suction(cup_targets[current[0]])
            return
        x, y = (event.x - border) / scale, (event.y - border) / scale
        if 0 <= x <= TABLE_WIDTH and 0 <= y <= TABLE_HEIGHT:
            simulation.move_to(x, y)

    def nudge(dx, dy):
        x, y = simulation.position
        simulation.move_to(x + dx, y + dy)

    held_keys: dict[int, str | None] = {}

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
        timer = held_keys.pop(event.keycode, None)
        if timer is not None:
            root.after_cancel(timer)

    def release_all(event=None):
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

    reset_button.configure(command=reset_match)
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
