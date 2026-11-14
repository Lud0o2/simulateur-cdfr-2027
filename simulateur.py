from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, isfinite, pi, radians, sin, sqrt
from pathlib import Path
from time import perf_counter


Point = tuple[float, float]
Rectangle = tuple[float, float, float, float]
TABLE_WIDTH = 3000
TABLE_HEIGHT = 2000
BORDER = 22
SIDE = 380
SPEED = 1000  # mm/s
TURN_STEP = radians(15)
TRIANGLE_HEIGHT = SIDE * sqrt(3) / 2
FOOTPRINT = (
    (-2 * TRIANGLE_HEIGHT / 3, 0),
    (TRIANGLE_HEIGHT / 3, -SIDE / 2),
    (TRIANGLE_HEIGHT / 3, SIDE / 2),
)
START = (250.0, 1000.0)
TABLE = ((0, 0), (TABLE_WIDTH, 0), (TABLE_WIDTH, TABLE_HEIGHT), (0, TABLE_HEIGHT))


def edges(points: tuple[Point, ...]):
    return zip(points, points[1:] + points[:1])


def obstacle_contact(
    position: Point, delta: Point, obstacle: Rectangle, footprint: tuple[Point, ...]
) -> float:
    """Renvoie la fraction du trajet où le triangle commence à pénétrer le rectangle."""
    x_min, y_min, x_max, y_max = obstacle
    corners = ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
    x, y = position
    dx, dy = delta
    enter, leave = 0.0, 1.0
    normals = tuple((by - ay, ax - bx) for (ax, ay), (bx, by) in edges(footprint))
    for ax, ay in ((1, 0), (0, 1), *normals):
        center = x * ax + y * ay
        offsets = [px * ax + py * ay for px, py in footprint]
        low, high = center + min(offsets), center + max(offsets)
        bounds = [px * ax + py * ay for px, py in corners]
        other_low, other_high = min(bounds), max(bounds)
        speed = dx * ax + dy * ay
        if speed == 0:
            if high <= other_low or low >= other_high:
                return 1.0
            continue
        first = (other_low - high) / speed
        last = (other_high - low) / speed
        enter = max(enter, min(first, last))
        leave = min(leave, max(first, last))
        if enter >= leave:
            return 1.0
    return enter


def arc_hits_edge(point: Point, angle: float, start: Point, end: Point) -> bool:
    """Teste l'intersection de l'arc d'un sommet avec un segment, autour de l'origine."""
    px, py = point
    sx, sy = start
    dx, dy = end[0] - sx, end[1] - sy
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return False
    closest = -(sx * dx + sy * dy) / length_squared
    nx, ny = sx + closest * dx, sy + closest * dy
    chord_squared = (px * px + py * py - nx * nx - ny * ny) / length_squared
    if chord_squared < 0:
        return False
    chord = sqrt(chord_squared)
    for fraction in (closest - chord, closest + chord):
        if 0 <= fraction <= 1:
            x, y = sx + fraction * dx, sy + fraction * dy
            swept = atan2(px * y - py * x, px * x + py * y)
            if 0 < swept <= angle or angle <= swept < 0:
                return True
    return False


@dataclass(frozen=True)
class Movement:
    start: Point
    end: Point
    started_at: float
    duration: float
    blocked: bool
    turn: bool = False


class Simulation:
    def __init__(self, obstacles: tuple[Rectangle, ...] = ()):
        self.obstacles = obstacles
        self.reset()

    def reset(self):
        self.time = 0.0
        self.heading = 0.0
        self.movement: Movement | None = None

    @property
    def footprint(self) -> tuple[Point, ...]:
        cosine, sine = cos(self.heading), sin(self.heading)
        return tuple((x * cosine - y * sine, x * sine + y * cosine) for x, y in FOOTPRINT)

    @property
    def moving(self) -> bool:
        move = self.movement
        return move is not None and self.time < move.started_at + move.duration

    @property
    def position(self) -> Point:
        move = self.movement
        if move is None:
            return START
        if not self.moving:
            return move.end
        fraction = (self.time - move.started_at) / move.duration
        x, y = move.start
        return x + (move.end[0] - x) * fraction, y + (move.end[1] - y) * fraction

    def move_to(self, x: float, y: float):
        if not (isfinite(x) and isfinite(y)):
            raise ValueError("La destination doit être finie.")
        start_x, start_y = self.position
        dx, dy = x - start_x, y - start_y
        distance = hypot(dx, dy)
        fraction = 1.0
        footprint = self.footprint
        xs, ys = zip(*footprint)
        if dx < 0:
            fraction = min(fraction, (-min(xs) - start_x) / dx)
        if dx > 0:
            fraction = min(fraction, (TABLE_WIDTH - max(xs) - start_x) / dx)
        if dy < 0:
            fraction = min(fraction, (-min(ys) - start_y) / dy)
        if dy > 0:
            fraction = min(fraction, (TABLE_HEIGHT - max(ys) - start_y) / dy)
        for obstacle in self.obstacles:
            fraction = min(fraction, obstacle_contact((start_x, start_y), (dx, dy), obstacle, footprint))

        blocked = distance > 0 and fraction < 1.0
        if blocked:
            # Un retrait microscopique évite un chevauchement dû aux arrondis.
            fraction = max(0.0, fraction - 1e-6 / distance)
            x, y = start_x + dx * fraction, start_y + dy * fraction
        self.movement = Movement((start_x, start_y), (x, y), self.time, distance * fraction / SPEED, blocked)

    def turn_by(self, angle: float):
        if not -pi < angle < pi:
            raise ValueError("La rotation doit être comprise entre -180° et 180°.")
        position = self.position
        heading = self.heading
        footprint = self.footprint
        self.heading = (self.heading + angle + pi) % (2 * pi) - pi
        blocked = self._turn_blocked(position, footprint, angle)
        if blocked:
            self.heading = heading
        self.movement = Movement(position, position, self.time, 0.0, blocked, turn=True)

    def _turn_blocked(self, position: Point, footprint: tuple[Point, ...], angle: float) -> bool:
        x, y = position
        destination = self.footprint
        if any(not (0 <= x + dx <= TABLE_WIDTH and 0 <= y + dy <= TABLE_HEIGHT) for dx, dy in destination):
            return True
        radius = 2 * TRIANGLE_HEIGHT / 3
        if min(x, y, TABLE_WIDTH - x, TABLE_HEIGHT - y) < radius:
            border = tuple((px - x, py - y) for px, py in TABLE)
            if any(arc_hits_edge(vertex, angle, a, b) for vertex in footprint for a, b in edges(border)):
                return True
        for obstacle in self.obstacles:
            left, top, right, bottom = obstacle
            if left > x + radius or right < x - radius or top > y + radius or bottom < y - radius:
                continue
            if obstacle_contact(position, (0, 0), obstacle, destination) < 1:
                return True
            corners = (
                (left - x, top - y), (right - x, top - y),
                (right - x, bottom - y), (left - x, bottom - y),
            )
            if any(arc_hits_edge(vertex, angle, a, b) for vertex in footprint for a, b in edges(corners)):
                return True
            # Dans le repère du robot, le rectangle tourne dans l'autre sens.
            if any(arc_hits_edge(vertex, -angle, a, b) for vertex in corners for a, b in edges(footprint)):
                return True
        return False

    def advance(self, seconds: float):
        if not isfinite(seconds) or seconds < 0:
            raise ValueError("La durée doit être finie et positive ou nulle.")
        self.time += seconds


def create_window(simulation: Simulation):
    import tkinter as tk

    background = "#151d24"
    foreground = "#e8eef3"
    muted = "#b5c4cf"
    accent = "#9cf2e1"
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
    tk.Button(header, text="Replacer au départ", command=simulation.reset).pack(side="right")

    mat = tk.PhotoImage(file=Path(__file__).with_name("tapis_2027.png"))
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
    path = canvas.create_line(0, 0, 0, 0, fill=accent, width=2, dash=(5, 4), state="hidden")
    robot = canvas.create_polygon(0, 0, 0, 0, 0, 0, fill="#152d3a", outline=accent, width=3)
    front = canvas.create_line(0, 0, 0, 0, fill=accent, width=3)

    footer = tk.Frame(root, bg=background)
    footer.pack(fill="x", pady=(8, 0))
    status = tk.Label(footer, bg=background, fg=foreground, font=("Segoe UI", 10, "bold"))
    status.pack(side="left")
    tk.Label(
        footer, text="Clic ou flèches · Q/D pour tourner", bg=background, fg=muted,
        font=("Segoe UI", 10),
    ).pack(side="right")

    def draw():
        x, y = simulation.position
        screen_x, screen_y = border + scale * x, border + scale * y
        vertices = [(screen_x + scale * dx, screen_y + scale * dy) for dx, dy in simulation.footprint]
        canvas.coords(robot, *(coordinate for vertex in vertices for coordinate in vertex))
        top, bottom = vertices[1:]
        middle_x, middle_y = (top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2
        mark_x = middle_x + (screen_x - middle_x) * 0.2
        mark_y = middle_y + (screen_y - middle_y) * 0.2
        mark_dx, mark_dy = (bottom[0] - top[0]) / 12, (bottom[1] - top[1]) / 12
        canvas.coords(front, mark_x - mark_dx, mark_y - mark_dy, mark_x + mark_dx, mark_y + mark_dy)
        move = simulation.movement
        if simulation.moving:
            end_x, end_y = move.end
            canvas.coords(path, screen_x, screen_y, border + scale * end_x, border + scale * end_y)
            canvas.itemconfigure(path, state="normal", fill="#ffcb84" if move.blocked else accent)
            label = "En déplacement"
        else:
            canvas.itemconfigure(path, state="hidden")
            label = "Au départ"
            if move is not None and move.turn:
                label = "Rotation bloquée" if move.blocked else "Cap réglé"
            elif move is not None:
                label = "Trajet bloqué" if move.blocked else "Arrivé"
        color = "#ffcb84" if move is not None and move.blocked else accent
        canvas.itemconfigure(robot, outline=color)
        status.configure(text=f"{label}  ·  X {x:.0f}  Y {y:.0f} mm  ·  {degrees(simulation.heading) % 360:.0f}°")

    def click(event):
        x, y = (event.x - border) / scale, (event.y - border) / scale
        if 0 <= x <= TABLE_WIDTH and 0 <= y <= TABLE_HEIGHT:
            simulation.move_to(x, y)

    def nudge(dx, dy):
        x, y = simulation.position
        simulation.move_to(x + dx, y + dy)

    held_keys: dict[int, str] = {}

    def press(event, action):
        keycode = event.keycode
        if keycode in held_keys:
            return

        def repeat():
            action()
            held_keys[keycode] = root.after(60, repeat)

        action()
        held_keys[keycode] = root.after(120, repeat)

    def release(event):
        timer = held_keys.pop(event.keycode, None)
        if timer is not None:
            root.after_cancel(timer)

    def release_all(event):
        for timer in held_keys.values():
            root.after_cancel(timer)
        held_keys.clear()

    canvas.bind("<Button-1>", click)
    root.bind("<KeyRelease>", release)
    root.bind("<FocusOut>", release_all)
    for key, dx, dy in (("Left", -100, 0), ("Right", 100, 0), ("Up", 0, -100), ("Down", 0, 100)):
        root.bind(f"<KeyPress-{key}>", lambda event, dx=dx, dy=dy: press(event, lambda: nudge(dx, dy)))
    for key, angle in (("q", -TURN_STEP), ("Q", -TURN_STEP), ("d", TURN_STEP), ("D", TURN_STEP)):
        root.bind(f"<KeyPress-{key}>", lambda event, angle=angle: press(event, lambda: simulation.turn_by(angle)))

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
    create_window(Simulation()).mainloop()
