"""Notification de fin de match ou d'entraînement."""

import os
from typing import Protocol


class Notificateur(Protocol):
    def fin(self) -> None: ...


class NotificateurBip:
    def fin(self) -> None:
        if os.name == "nt":
            import winsound

            try:
                winsound.Beep(880, 180)
                return
            except RuntimeError:
                pass
        print("\a", end="", flush=True)


class NotificateurSilencieux:
    def fin(self) -> None:
        return None


def completion_beep() -> None:
    NotificateurBip().fin()
