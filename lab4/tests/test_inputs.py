import pytest

from criteria import BayesCriterion
from model import Instrument, StageParams
from solver import BellmanSolver, TransitionModel


def asset(name="A", package=10.0, floor=0.0, fee=0.0):
    return Instrument(name=name, package=package, min_volume=floor, commission=fee)


class TestInvalidInputs:
    def test_probs_must_sum_to_one(self):
        stages = (StageParams((0.5, 0.4), ((1.1,), (1.0,))),)
        with pytest.raises(ValueError):
            TransitionModel((asset(),), stages, (0.0, 10.0))

    def test_commission_must_be_below_one(self):
        with pytest.raises(ValueError):
            TransitionModel((asset(fee=1.0),), (StageParams((1.0,), ((1.0,),)),), (0.0, 10.0))

    def test_negative_start_state(self):
        with pytest.raises(ValueError):
            TransitionModel((asset(),), (StageParams((1.0,), ((1.0,),)),), (-5.0, 10.0))

    def test_negative_min_volume(self):
        with pytest.raises(ValueError):
            TransitionModel((asset(floor=-1.0),), (StageParams((1.0,), ((1.0,),)),), (0.0, 10.0))

    def test_bad_package(self):
        with pytest.raises(ValueError):
            TransitionModel((asset(package=0.0),), (StageParams((1.0,), ((1.0,),)),), (0.0, 10.0))

    def test_infeasible_start_raises_runtime(self):
        tight = asset(package=10.0, floor=100.0)
        stages = (StageParams((1.0,), ((1.05,),)),)
        with pytest.raises(RuntimeError):
            BellmanSolver(TransitionModel((tight,), stages, (10.0, 0.0))).solve(BayesCriterion())


class TestZeroData:
    def test_zero_capital_no_trades_zero_income(self):
        stages = (StageParams((1.0,), ((1.1,),)),)
        sol = BellmanSolver(TransitionModel((asset(),), stages, (0.0, 0.0))).solve(BayesCriterion())
        assert sol.root_value == pytest.approx(0.0)
        assert sol.paths[0].controls[0] == (0,)

    def test_zero_growth_commissions_kill_trading(self):
        stages = tuple(StageParams((1.0, 0.0), ((1.0,), (1.0,))) for _ in range(2))
        sol = BellmanSolver(TransitionModel((asset(fee=0.05),), stages, (50.0, 50.0))).solve(BayesCriterion())
        assert sol.root_value == pytest.approx(0.0)
        for path in sol.paths:
            assert path.controls[0] == (0,)


class TestAllResourceToOneProject:
    def test_dominant_asset_takes_all_cash(self):
        weak = asset("Слабый", package=10.0, fee=0.0)
        strong = asset("Сильный", package=10.0, fee=0.0)
        stages = (StageParams((1.0,), ((1.01, 2.0),)),)
        tm = TransitionModel((weak, strong), stages, (0.0, 0.0, 60.0))
        sol = BellmanSolver(tm).solve(BayesCriterion())
        assert sol.paths[0].controls[0] == (0, 6)
        assert sol.root_value == pytest.approx(60.0)
        vals = {}
        for row in tm.enumerate_controls((0.0, 0.0, 60.0)):
            row = tuple(int(x) for x in row)
            after, cash = tm.apply_control((0.0, 0.0, 60.0), row)
            vals[row] = after[0] * 1.01 + after[1] * 2.0 + cash - 60.0
        best = max(vals.values())
        assert vals[(0, 6)] == pytest.approx(best)
        assert all(v <= best + 1e-9 for v in vals.values())


class TestMultipleOptima:
    def test_symmetric_assets_alternative_optima(self):
        left = asset("L", package=10.0, fee=0.0)
        right = asset("R", package=10.0, fee=0.0)
        stages = (StageParams((1.0,), ((2.0, 2.0),)),)
        tm = TransitionModel((left, right), stages, (10.0, 10.0, 10.0))
        sol = BellmanSolver(tm).solve(BayesCriterion())
        u = sol.paths[0].controls[0]
        vals = {}
        for row in tm.enumerate_controls((10.0, 10.0, 10.0)):
            row = tuple(int(x) for x in row)
            after, cash = tm.apply_control((10.0, 10.0, 10.0), row)
            vals[row] = (after[0] + after[1]) * 2.0 + cash - 30.0
        best = max(vals.values())
        assert vals[u] == pytest.approx(best)
        assert sum(1 for v in vals.values() if v == pytest.approx(best)) >= 2


class TestBruteForce:
    def step(self, tm, stages, state, u, stage, s):
        w = tm.stage_income(state, u, stage, s)
        after, cash = tm.apply_control(state, u)
        nxt = (after[0] * stages[stage].coeffs[s][0], cash)
        return w, nxt

    def test_two_stage_brute_force_matches_dp_bayes(self):
        stages = (StageParams((0.5, 0.5), ((2.0,), (1.0,))),
                  StageParams((0.5, 0.5), ((2.0,), (1.0,))))
        tm = TransitionModel((asset(package=10.0),), stages, (10.0, 20.0))
        sol = BellmanSolver(tm).solve(BayesCriterion())
        start = (10.0, 20.0)
        best = -1e18
        for row0 in tm.enumerate_controls(start):
            u0 = tuple(int(x) for x in row0)
            total = 0.0
            for s in (0, 1):
                w0, nxt = self.step(tm, stages, start, u0, 0, s)
                last = -1e18
                for row1 in tm.enumerate_controls(nxt):
                    u1 = tuple(int(x) for x in row1)
                    mean = 0.5 * (tm.stage_income(nxt, u1, 1, 0) + tm.stage_income(nxt, u1, 1, 1))
                    last = max(last, mean)
                total += 0.5 * (w0 + last)
            best = max(best, total)
        assert sol.root_value == pytest.approx(best, abs=1e-9)
