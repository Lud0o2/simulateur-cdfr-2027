"""Métriques de comparaison des matchs."""


def _match_rank(score, duration):
    """Classe un match par points par seconde, puis par total de points."""
    return score / max(duration, 1e-6), score


def _replay_rank(replay):
    return replay["score"], -replay["duree"]


def _format_elapsed(seconds):
    hours, rest = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {seconds:02d} s"
    return f"{seconds} s"
