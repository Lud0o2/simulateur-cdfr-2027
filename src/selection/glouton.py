"""Sélection gloutonne par gain divisé par durée."""


class Glouton:
    id = "glouton"

    def classer(self, observation, candidats):
        return sorted(candidats, key=lambda candidat: candidat.gain / max(candidat.duration_s, 1e-9),
                      reverse=True)

    def observer(self, candidat, valeur: float) -> None:
        return None

    def order(self, simulation, held, choices, *, pickup, can_finish=False, pickup_plans=None):
        if pickup:
            return sorted(choices, key=lambda choice: (choice[0], choice[1], choice[2]))
        return sorted(choices, key=lambda choice: (-(choice[0] / max(choice[1], 1e-9)), choice[2]))

    def observe(self, choice):
        return None
