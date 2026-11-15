"""Répétition des commandes clavier du pilotage manuel."""


class RepeteurClavier:
    def __init__(self, root, *, premier_delai: int = 120, delai: int = 60):
        self.root = root
        self.premier_delai = premier_delai
        self.delai = delai
        self.held = {}

    def press(self, event, action, *, repeats=True):
        keycode = event.keycode
        if keycode in self.held:
            return

        def repeat():
            action()
            self.held[keycode] = self.root.after(self.delai, repeat)

        action()
        self.held[keycode] = self.root.after(self.premier_delai, repeat) if repeats else None

    def release(self, event):
        timer = self.held.pop(event.keycode, None)
        if timer is not None:
            self.root.after_cancel(timer)

    def release_all(self, event=None):
        for timer in self.held.values():
            if timer is not None:
                self.root.after_cancel(timer)
        self.held.clear()


class ControleurManuel:
    """Clics et déplacements clavier du pilotage manuel."""

    def __init__(self, canvas, simulation, *, cup_targets, etat):
        self.canvas = canvas
        self.simulation = simulation
        self.cup_targets = cup_targets
        self.etat = etat

    def click(self, event):
        scale, border = self.etat()
        if not scale:
            return
        current = self.canvas.find_withtag("current")
        if current and current[0] in self.cup_targets:
            self.simulation.toggle_suction(self.cup_targets[current[0]])
            return
        from ..terrains.officiel_2027 import TABLE_HEIGHT, TABLE_WIDTH

        x, y = (event.x - border) / scale, (event.y - border) / scale
        if 0 <= x <= TABLE_WIDTH and 0 <= y <= TABLE_HEIGHT:
            self.simulation.move_to(x, y)

    def nudge(self, dx, dy):
        x, y = self.simulation.position
        self.simulation.move_to(x + dx, y + dy)
