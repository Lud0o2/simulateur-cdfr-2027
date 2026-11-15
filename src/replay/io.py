"""Sauvegarde atomique des replays."""

import json
from itertools import count
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..infra.ressources import REPLAY_DIR
from .store import _trials


def save_replay(replay, directory: Path = REPLAY_DIR, *, descriptive=False, trials=None):
    suffix = f"-{replay['score']:d}pts-{replay['duree']:.2f}s" if descriptive else ""
    directory.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(dir=directory, prefix=".essai-", suffix=".tmp", encoding="utf-8",
                              mode="w", delete=False)
    temporary = Path(file.name)
    try:
        with file:
            json.dump(replay, file, ensure_ascii=False, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        for number in count(max((_trials if trials is None else trials)(directory), default=0) + 1):
            stem = f"essai-{number:04d}"
            reservation = directory / f".{stem}.lock"
            try:
                reservation.mkdir()
            except FileExistsError:
                continue
            try:
                # Deux sauvegardes peuvent viser le même numéro avec des noms différents.
                if (directory / f"{stem}.json").exists() or any(directory.glob(f"{stem}-*.json")):
                    continue
                path = directory / f"{stem}{suffix}.json"
                # Sous Windows, rename échoue si la destination existe. Sous POSIX, le lien offre cette garantie.
                if os.name == "nt":
                    temporary.rename(path)
                else:
                    path.hardlink_to(temporary)
                return number, path
            except FileExistsError:
                continue
            finally:
                reservation.rmdir()
    finally:
        temporary.unlink(missing_ok=True)


