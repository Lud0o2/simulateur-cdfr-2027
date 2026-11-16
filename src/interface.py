"""Façade de compatibilité de l'interface Tkinter."""

from .replay.player import Playback
from .ui.window import create_window

__all__ = ["Playback", "create_window"]


if __name__ == "__main__":
    from .physique.moteur import Simulation

    try:
        simulation = Simulation()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    create_window(simulation).mainloop()
