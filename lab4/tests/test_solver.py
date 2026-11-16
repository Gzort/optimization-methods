import math

import numpy as np
import pytest

from criteria import BayesCriterion
from model import Instrument, StageParams, XI1_VALUES, STAGES, INSTRUMENTS
from solver import BellmanSolver, TransitionModel

ASSET = Instrument("А", package=10.0, min_volume=0.0, commission=0.0)
ONE_STAGE = (StageParams(probs=(0.5, 0.5), coeffs=((2.0,), (1.0,))),)
TWO_STAGES = (StageParams(probs=(0.5, 0.5), coeffs=((2.0,), (1.0,))),
              StageParams(probs=(0.5, 0.5), coeffs=((2.0,), (1.0,))))

def make_tm(stages, xi1):
    return TransitionModel(instruments=(ASSET,), stages=stages, xi1_values=xi1)

class TestMiniOneStage:
    def test_bayes_optimal(self):
        sol = BellmanSolver(make_tm(ONE_STAGE, (0.0, 20.0))).solve(BayesCriterion())
        assert sol.root_value == pytest.approx(10.0)
        assert sol.paths[0].controls[0] == (2,)


class TestMiniTwoStages:
    def test_bayes_two_stage_hand_value(self):
        sol = BellmanSolver(make_tm(TWO_STAGES, (0.0, 20.0))).solve(BayesCriterion())
        assert sol.root_value == pytest.approx(25.0)
        assert sol.paths[0].controls[0] == (2,)

    def test_bayes_expected_income_equals_root_value(self):
        sol = BellmanSolver(make_tm(TWO_STAGES, (0.0, 20.0))).solve(BayesCriterion())
        assert sol.expected_income == pytest.approx(sol.root_value, abs=1e-9)

class TestReducedTaskInvariants:

    def test_all_9_paths_present(self, reduced_bayes_solution):
        assert len(reduced_bayes_solution.paths) == 9
        assert sum(p.prob for p in reduced_bayes_solution.paths) == pytest.approx(1.0)

    def test_bayes_root_equals_expected_path_income(self, reduced_bayes_solution):
        assert reduced_bayes_solution.expected_income == pytest.approx(
            reduced_bayes_solution.root_value, abs=1e-9)

    def test_policy_feasible_along_all_paths(self, reduced_tm, reduced_bayes_solution):
        for path in reduced_bayes_solution.paths:
            for stage, (values, u) in enumerate(zip(path.states, path.controls)):
                assert reduced_tm.apply_control(values, u) is not None, (path.scenarios, stage)

    def test_income_telescopes_to_wealth_change(self, reduced_bayes_solution):
        w1 = sum(XI1_VALUES)
        for path in reduced_bayes_solution.paths:
            assert sum(path.incomes) == pytest.approx(path.total_income, abs=1e-9)
            assert path.final_wealth == pytest.approx(w1 + path.total_income, abs=1e-9)

    def test_bayes_beats_do_nothing_baseline(self, reduced_tm, reduced_bayes_solution):
        baseline = BellmanSolver(reduced_tm, XI1_VALUES).solve(BayesCriterion(), force_zero=True)
        assert reduced_bayes_solution.expected_income >= baseline.expected_income - 1e-9
