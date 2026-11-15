"""Sources aléatoires reproductibles."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceAleatoire:
    graine: int | None = None

    def generateur(self) -> np.random.Generator:
        return np.random.default_rng(self.graine)

    def suite(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(self.graine)
