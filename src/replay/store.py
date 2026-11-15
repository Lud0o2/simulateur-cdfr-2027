"""Index des essais enregistrés."""

from pathlib import Path
import re


def _trials(directory: Path):
    trials = {}
    for path in directory.glob("essai-*.json"):
        match = re.fullmatch(r"essai-([0-9]+)(?:--?[0-9]+pts-[0-9]+\.[0-9]{2}s)?", path.stem)
        if match and path.is_file():
            trials[int(match[1])] = path
    return trials


