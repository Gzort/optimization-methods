from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from model import Instrument, StageParams

EPS = 1e-9

def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))

class TransitionModel:

    def __init__(self,
                 instruments: Sequence[Instrument],
                 stages: Sequence[StageParams],
                 cash_grid: float = 5.0):
        self.instruments = tuple(instruments)
        self.stages = tuple(stages)
        self.cash_grid = float(cash_grid)
        self.grids = tuple(inst.grid for inst in self.instruments) + (self.cash_grid,)
        self.packages = tuple(inst.package for inst in self.instruments)
        self.mins = tuple(inst.min_volume for inst in self.instruments)
        self.commissions = tuple(inst.commission for inst in self.instruments)

    def values_to_units(self, state_values: Sequence[float]) -> tuple[int, ...]:
        return tuple(_round_half_up(v / g) for v, g in zip(state_values, self.grids))

    def units_to_values(self, state_units: Sequence[int]) -> tuple[float, ...]:
        return tuple(u * g for u, g in zip(state_units, self.grids))

    def state_key(self, state_units: Sequence[int]) -> bytes:
        return np.asarray(state_units, dtype=np.int64).tobytes()

    @staticmethod
    def _unique_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = np.ascontiguousarray(rows).view([("", np.int64)] * rows.shape[1]).ravel()
        uniq, inverse = np.unique(flat, return_inverse=True)
        return uniq.view(np.int64).reshape(-1, rows.shape[1]), inverse

    def cash_flow(self, u: int, instrument: Instrument) -> float:
        h, c = instrument.package, instrument.commission
        rate = (1.0 + c) if u > 0 else (1.0 - c)
        return u * h * rate

    def apply_control(self,
                      state_values: Sequence[float],
                      u: Sequence[int]) -> Optional[tuple[tuple[float, ...], float]]:
        S = state_values[:-1]
        C = state_values[-1]
        after = tuple(s + ui * inst.package for s, ui, inst in zip(S, u, self.instruments))
        cash_after = C - sum(self.cash_flow(ui, inst) for ui, inst in zip(u, self.instruments))
        if cash_after < -EPS:
            return None
        if any(a < inst.min_volume - EPS for a, inst in zip(after, self.instruments)):
            return None
        return after, cash_after

    def grow_and_round(self,
                       asset_values: Sequence[float],
                       cash: float,
                       stage: int,
                       scenario: int) -> tuple[int, ...]:
        coeffs = self.stages[stage].coeffs[scenario]
        grown = (a * k for a, k in zip(asset_values, coeffs))
        return self.values_to_units((*grown, cash))

    def stage_income(self,
                     state_values: Sequence[float],
                     u: Sequence[int],
                     stage: int,
                     scenario: int) -> float:
        applied = self.apply_control(state_values, u)
        if applied is None:
            raise ValueError(f"Недопустимое управление {u} для состояния {state_values}")
        after, cash_after = applied
        next_units = self.grow_and_round(after, cash_after, stage, scenario)
        wealth_before = sum(state_values)
        wealth_after = sum(self.units_to_values(next_units))
        return wealth_after - wealth_before

    def cash_value(self, state_units: Sequence[int]) -> float:
        return state_units[-1] * self.cash_grid

    def control_bounds(self, state_units: Sequence[int]) -> Optional[tuple[tuple[int, int], ...]]:
        S, C = self.units_to_values(state_units[:-1]), self.cash_value(state_units)
        bounds = []
        for s, inst in zip(S, self.instruments):
            lo = math.ceil((inst.min_volume - s) / inst.package - EPS)
            hi = math.floor(C / (inst.package * (1.0 + inst.commission)) + EPS)
            if lo > hi:
                return None
            bounds.append((lo, hi))
        return tuple(bounds)

    def enumerate_controls(self, state_units: Sequence[int]) -> np.ndarray:
        bounds = self.control_bounds(state_units)
        if bounds is None:
            return np.zeros((0, 3), dtype=np.int64)
        axes = [np.arange(lo, hi + 1, dtype=np.int64) for lo, hi in bounds]
        grid = np.meshgrid(*axes, indexing="ij")
        return np.stack([g.ravel() for g in grid], axis=1)

    def stage_transitions(self,
                          stage: int,
                          state_units: Sequence[int],
                          need_next: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_scen = len(self.stages[stage].probs)
        U_all = self.enumerate_controls(state_units)
        if U_all.shape[0] == 0:
            empty_f = np.zeros((0, n_scen))
            return U_all, empty_f, np.zeros((0, n_scen, len(self.grids)), dtype=np.int64)

        S = np.array(self.units_to_values(state_units[:-1]))
        C = self.cash_value(state_units)
        h = np.array(self.packages)
        c = np.array(self.commissions)
        mins = np.array(self.mins)

        U = U_all.astype(float)
        flow = np.where(U > 0, U * h * (1.0 + c), U * h * (1.0 - c))
        cash_after = C - flow.sum(axis=1)
        volumes = S[None, :] + U * h[None, :]
        feasible = (cash_after >= -EPS) & (volumes >= mins[None, :] - EPS).all(axis=1)
        U, cash_after, volumes = U_all[feasible], cash_after[feasible], volumes[feasible]

        wealth_before = S.sum() + C
        F = np.empty((U.shape[0], n_scen))
        Next = np.zeros((U.shape[0], n_scen, len(self.grids)), dtype=np.int64)
        grid_arr = np.array(self.grids)
        for s in range(n_scen):
            coeffs = np.array(self.stages[stage].coeffs[s])
            grown = volumes * coeffs[None, :]
            with_cash = np.concatenate([grown, cash_after[:, None]], axis=1)
            units = np.floor(with_cash / grid_arr[None, :] + 0.5).astype(np.int64)
            F[:, s] = (units * grid_arr[None, :]).sum(axis=1) - wealth_before
            if need_next:
                Next[:, s, :] = units
        return U, F, Next

NEG_INF = float("-inf")

@dataclass(frozen=True)
class PathRecord:

    scenarios: tuple[int, ...]
    prob: float
    states_units: tuple[tuple[int, ...], ...]
    controls: tuple[tuple[int, ...], ...]
    incomes: tuple[float, ...]
    total_income: float
    final_wealth: float

@dataclass
class Solution:

    criterion_name: str
    root_value: float
    xi1_units: tuple[int, ...]
    paths: list[PathRecord] = field(default_factory=list)
    n_states: dict[int, int] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    @property
    def expected_income(self) -> float:
        return sum(p.prob * p.total_income for p in self.paths)

    @property
    def worst_income(self) -> float:
        return min(p.total_income for p in self.paths)

    @property
    def best_income(self) -> float:
        return max(p.total_income for p in self.paths)

class BellmanSolver:

    def __init__(self, model: TransitionModel, xi1_values: Sequence[float]):
        self.tm = model
        self.xi1_units = model.values_to_units(xi1_values)
        self._value: dict[tuple[int, int], float] = {}
        self._best_u: dict[tuple[int, int], Optional[tuple[int, ...]]] = {}
        self._criterion = None
        self._force_zero = False

    def solve(self, criterion, force_zero: bool = False) -> Solution:
        self._value.clear()
        self._best_u.clear()
        self._criterion = criterion
        self._force_zero = force_zero
        t0 = time.perf_counter()
        root = self._bellman(0, self.xi1_units)
        if root == NEG_INF:
            raise RuntimeError("Из начального состояния нет допустимых управлений")
        paths = self._forward_pass()
        n_states = dict(Counter(stage for stage, _ in self._value))
        return Solution(criterion_name=criterion.name, root_value=root,
                        xi1_units=self.xi1_units, paths=paths, n_states=n_states,
                        elapsed_sec=time.perf_counter() - t0)

    def _bellman(self, stage: int, state_units: tuple[int, ...]) -> float:
        last = stage == len(self.tm.stages) - 1
        key = (stage, self.tm.state_key(state_units))
        cached = self._value.get(key)
        if cached is not None:
            return cached

        U, F, Next = self.tm.stage_transitions(stage, state_units, need_next=not last)
        if self._force_zero:
            zero_rows = np.where((U == 0).all(axis=1))[0]
            if len(zero_rows):
                U, F, Next = U[zero_rows], F[zero_rows], Next[zero_rows]
        if U.shape[0] == 0:
            self._value[key], self._best_u[key] = NEG_INF, None
            return NEG_INF

        if last:
            total = F
        else:
            total = np.empty_like(F)
            for s in range(F.shape[1]):
                uniq_rows, inv = self.tm._unique_rows(Next[:, s, :])
                future = np.array([self._bellman(stage + 1, tuple(int(x) for x in row))
                                   for row in uniq_rows])
                total[:, s] = F[:, s] + future[inv]

        probs = np.array(self.tm.stages[stage].probs)
        scores = self._criterion.scores(total, probs)
        best = int(np.argmax(scores))
        self._value[key] = float(scores[best])
        self._best_u[key] = tuple(int(x) for x in U[best])
        return self._value[key]

    def _forward_pass(self) -> list[PathRecord]:
        paths: list[PathRecord] = []
        n_stages = len(self.tm.stages)
        n_scen = len(self.tm.stages[0].probs)

        def walk(stage: int, state_units: tuple[int, ...], scen_acc: tuple[int, ...],
                 prob: float, states: tuple, controls: tuple, incomes: tuple) -> None:
            if stage == n_stages:
                total = float(sum(incomes))
                paths.append(PathRecord(scenarios=scen_acc, prob=prob,
                                        states_units=states, controls=controls,
                                        incomes=incomes, total_income=total,
                                        final_wealth=total + sum(self.tm.units_to_values(self.xi1_units))))
                return
            u = self._best_u[(stage, self.tm.state_key(state_units))]
            values = self.tm.units_to_values(state_units)
            after, cash_after = self.tm.apply_control(values, u)
            for s in range(n_scen):
                w = self.tm.stage_income(values, u, stage, s)
                nxt = self.tm.grow_and_round(after, cash_after, stage, s)
                walk(stage + 1, nxt, scen_acc + (s,),
                     prob * self.tm.stages[stage].probs[s],
                     states + (state_units,), controls + (u,), incomes + (w,))

        walk(0, self.xi1_units, (), 1.0, (), (), ())
        return paths
