import numpy as np

class Criterion:

    name: str = "Критерий"

    def scores(self, F: np.ndarray, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class BayesCriterion(Criterion):

    name = "Байес"

    def scores(self, F: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return F @ np.asarray(probs, dtype=float)

class LaplaceCriterion(Criterion):

    name = "Лаплас"

    def scores(self, F: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return F.mean(axis=1)

class WaldCriterion(Criterion):

    name = "Вальд"

    def scores(self, F: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return F.min(axis=1)

class HurwiczCriterion(Criterion):

    def __init__(self, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Коэффициент оптимизма alpha должен быть в [0, 1], получено {alpha}")
        self.alpha = float(alpha)
        self.name = f"Гурвиц (alpha={self.alpha:g})"

    def scores(self, F: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self.alpha * F.max(axis=1) + (1.0 - self.alpha) * F.min(axis=1)

class SavageCriterion(Criterion):

    name = "Сэвидж"

    def scores(self, F: np.ndarray, probs: np.ndarray) -> np.ndarray:
        best_per_scenario = F.max(axis=0)
        regrets = best_per_scenario[None, :] - F
        return -regrets.max(axis=1)

