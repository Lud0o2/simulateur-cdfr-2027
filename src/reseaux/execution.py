"""Exécution série, processus ou factice des matchs d'entraînement."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext


class ExecuteurMatch:
    def __init__(self, workers: int = 1, *, executor=None):
        self.workers = workers
        self.executor = executor

    def executer(self, fonction, iterable):
        if self.workers <= 1:
            return map(fonction, iterable)
        if self.executor is not None:
            return self.executor.map(fonction, iterable)
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(fonction, iterable))
