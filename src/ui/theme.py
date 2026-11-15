"""Thème visuel de l'interface."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    background: str = "#111820"
    surface: str = "#1b2732"
    border_color: str = "#344450"
    foreground: str = "#f0f4f6"
    muted: str = "#aabac5"
    accent: str = "#84e0cb"
    yellow: str = "#f5d477"
    warning: str = "#ffbc75"
    font: str = "Segoe UI"
