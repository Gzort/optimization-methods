import numpy as np


class BayesCriterion:

    name = "Байес"

    def scores(self, F: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return F @ np.asarray(probs, dtype=float)
