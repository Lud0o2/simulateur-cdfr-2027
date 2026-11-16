"""Validation et lecture d'un replay."""

from bisect import bisect_right

from ..domaine.validation import _finite_number


class Replay:
    def __init__(self, data):
        def coordinates(values, count):
            return (isinstance(values, (list, tuple)) and len(values) == count
                    and all(map(_finite_number, values)))

        def polygon(values, minimum):
            return (isinstance(values, list) and len(values) >= minimum
                    and all(coordinates(point, 2) for point in values))

        if not isinstance(data, dict) or not isinstance(data.get("frames"), list) or not data["frames"]:
            raise ValueError("le replay doit contenir au moins une image")
        obstacles = data.get("obstacles")
        if not isinstance(obstacles, (list, tuple)) or not all(coordinates(rect, 4) for rect in obstacles):
            raise ValueError("liste d’obstacles manquante ou invalide")
        self.frames = data["frames"]
        self.obstacles = obstacles
        self.collection = data.get("strategie") == "opti_take"
        self.times = []
        for index, frame in enumerate(self.frames, 1):
            if not isinstance(frame, dict):
                raise ValueError(f"image {index} invalide")
            for key in ("t", "position", "heading", "robot", "piles", "cups", "target", "blocked", "score", "action"):
                if key not in frame:
                    raise ValueError(f"image {index} : champ « {key} » manquant")

            timestamp = frame["t"]
            if not _finite_number(timestamp) or timestamp < 0 or (self.times and timestamp <= self.times[-1]):
                raise ValueError(f"image {index} : horodatage invalide")
            if not (coordinates(frame["position"], 2) and _finite_number(frame["heading"])
                    and (frame["target"] is None or coordinates(frame["target"], 2))
                    and polygon(frame["robot"], 3)
                    and isinstance(frame["piles"], list) and all(polygon(pile, 4) for pile in frame["piles"])
                    and isinstance(frame["cups"], list) and all(
                        isinstance(cup, list) and len(cup) == 2 and coordinates(cup[0], 2) and type(cup[1]) is bool
                        for cup in frame["cups"]
                    )
                    and type(frame["blocked"]) is bool and type(frame["score"]) is int
                    and isinstance(frame["action"], str)):
                raise ValueError(f"image {index} : données d’affichage invalides")
            if index > 1:
                previous = self.frames[index - 2]
                unloaded = frame["score"] - previous["score"] if self.collection else 0
                if (unloaded < 0 or len(frame["piles"]) != len(previous["piles"]) - unloaded
                        or len(frame["cups"]) != len(previous["cups"])):
                    raise ValueError(f"image {index} : nombre de piles ou de ventouses incohérent")
            self.times.append(timestamp)
        self.start = self.time = self.times[0]
        self.duration = self.times[-1]

    @property
    def frame(self):
        return self.frames[max(0, bisect_right(self.times, self.time) - 1)]


