"""Façade d'entraînement ; la boucle vit encore dans `src.reseau`."""


def train_matches(*args, **kwargs):
    from ..reseau import train_matches as run

    return run(*args, **kwargs)


def _train_matches(*args, **kwargs):
    from ..reseau import _train_matches as run

    return run(*args, **kwargs)


def _train_endless(*args, **kwargs):
    from ..reseau import _train_endless as run

    return run(*args, **kwargs)
