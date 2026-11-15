"""Décorateur d'exploration epsilon-gourmand."""


class EpsilonGlouton:
    def __init__(self, selecteur, rng, epsilon: float = 0.1):
        self.selecteur = selecteur
        self.rng = rng
        self.epsilon = epsilon
        self.id = f"epsilon_{selecteur.id}"

    def classer(self, observation, candidats):
        if candidats and self.rng.random() < self.epsilon:
            ordonnes = list(candidats)
            self.rng.shuffle(ordonnes)
            return ordonnes
        return self.selecteur.classer(observation, candidats)

    def observer(self, candidat, valeur: float) -> None:
        self.selecteur.observer(candidat, valeur)
