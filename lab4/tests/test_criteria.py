import numpy as np

from criteria import BayesCriterion

F = np.array([[8.0, 6.0, 4.0],
              [7.0, 7.0, 7.0],
              [5.0, 9.0, 10.0]])
P = np.array([0.5, 0.3, 0.2])

def test_bayes_scores():
    scores = BayesCriterion().scores(F, P)
    assert np.allclose(scores, [6.6, 7.0, 7.2])

def test_bayes_best_control_on_reference_matrix():
    assert int(np.argmax(BayesCriterion().scores(F, P))) == 2

def test_bayes_ignores_row_scaling_symmetry():
    shifted = F + 100.0
    assert np.allclose(BayesCriterion().scores(shifted, P) - 100.0,
                       BayesCriterion().scores(F, P))
