import pytest

from criteria import BayesCriterion
from model import INSTRUMENTS, STAGES, XI1_VALUES
from solver import BellmanSolver, TransitionModel


@pytest.fixture(scope="session")
def reduced_tm() -> TransitionModel:
    return TransitionModel(INSTRUMENTS, STAGES[:2], XI1_VALUES)


@pytest.fixture(scope="session")
def reduced_bayes_solution(reduced_tm):
    return BellmanSolver(reduced_tm).solve(BayesCriterion())
