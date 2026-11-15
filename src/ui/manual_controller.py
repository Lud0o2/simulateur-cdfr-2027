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
