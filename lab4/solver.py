from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from model import Instrument, StageParams, validate_problem

EPS = 1e-9
PROGRESS_EVERY = 50


class TransitionModel:

    def __init__(self,
                 instruments: Sequence[Instrument],
                 stages: Sequence[StageParams],
                 xi1_values: Sequence[float]):
        validate_problem(instruments, stages, xi1_values)
        self.instruments = tuple(instruments)
        self.stages = tuple(stages)
        self.xi1_values = tuple(float(v) for v in xi1_values)
        self.n_assets = len(self.instruments)
        self.n_scen = len(self.stages[0].probs)
        self.n_stages = len(self.stages)
        self.packages = tuple(inst.package for inst in self.instruments)
        self.mins = tuple(inst.min_volume for inst in self.instruments)
        self.commissions = tuple(inst.commission for inst in self.instruments)
        self._h = np.array(self.packages)
        self._c = np.array(self.commissions)
        self._mins = np.array(self.mins)

    def cash_flow(self, u: int, instrument: Instrument) -> float:
        rate = (1.0 + instrument.commission) if u > 0 else (1.0 - instrument.commission)
        return u * instrument.package * rate

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

    def control_bounds(self, state_values: Sequence[float]) -> Optional[tuple[tuple[int, int], ...]]:
        S = state_values[:-1]
        C = state_values[-1]
        budget = C + sum((s - inst.min_volume) * (1.0 - inst.commission)
                         for s, inst in zip(S, self.instruments))
        bounds = []
        for j, inst in enumerate(self.instruments):
            lo = math.ceil((inst.min_volume - S[j]) / inst.package - EPS)
            hi = math.floor(budget / (inst.package * (1.0 + inst.commission)) + EPS)
            if lo > hi:
                return None
            bounds.append((lo, hi))
        return tuple(bounds)

    def enumerate_controls(self, state_values: Sequence[float]) -> np.ndarray:
        bounds = self.control_bounds(state_values)
        if bounds is None:
            return np.zeros((0, self.n_assets), dtype=np.int64)
        if self.n_assets == 3:
            return self._enumerate_three(bounds, state_values)
        return self._enumerate_generic(bounds, state_values)

    def _enumerate_generic(self, bounds, state_values: Sequence[float]) -> np.ndarray:
        axes = [np.arange(lo, hi + 1, dtype=np.int64) for lo, hi in bounds]
        if any(a.size == 0 for a in axes):
            return np.zeros((0, self.n_assets), dtype=np.int64)
        mesh = np.meshgrid(*axes, indexing="ij")
        U = np.stack([m.ravel() for m in mesh], axis=1)
        floats = U.astype(float)
        flow = np.where(floats > 0, floats * self._h * (1.0 + self._c),
                        floats * self._h * (1.0 - self._c)).sum(axis=1)
        return U[flow <= state_values[-1] + EPS]

    def _enumerate_three(self, bounds, state_values: Sequence[float]) -> np.ndarray:
        (lo1, _), (lo2, hi2), (lo3, hi3) = bounds
        C = state_values[-1]
        h1, h2, h3 = self.packages
        c1, c2, c3 = self.commissions

        u2v = np.arange(lo2, hi2 + 1, dtype=np.int64)
        u3v = np.arange(lo3, hi3 + 1, dtype=np.int64)
        flow2 = np.where(u2v > 0, u2v * h2 * (1.0 + c2), u2v * h2 * (1.0 - c2))
        flow3 = np.where(u3v > 0, u3v * h3 * (1.0 + c3), u3v * h3 * (1.0 - c3))
        residual = C - flow2[:, None] - flow3[None, :]
        step1 = np.where(residual >= 0.0, h1 * (1.0 + c1), h1 * (1.0 - c1))
        u1hi = np.floor(residual / step1 + EPS).astype(np.int64)
        lens = np.maximum(u1hi - lo1 + 1, 0)

        flat = lens.ravel()
        total = int(flat.sum())
        if total == 0:
            return np.zeros((0, 3), dtype=np.int64)
        ends = np.cumsum(flat)
        starts = ends - flat
        idx = np.arange(total, dtype=np.int64) - np.repeat(starts, flat)
        u1 = lo1 + idx
        u2 = np.repeat(np.repeat(u2v, len(u3v)), flat)
        u3 = np.repeat(np.tile(u3v, len(u2v)), flat)
        return np.stack([u1, u2, u3], axis=1)

    def stage_transitions(self,
                          stage: int,
                          state_values: Sequence[float],
                          need_next: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U = self.enumerate_controls(state_values)
        K = U.shape[0]
        if K == 0:
            return U, np.zeros((0, self.n_scen)), np.zeros((0, self.n_scen, self.n_assets + 1))

        S = np.array(state_values[:-1])
        C = state_values[-1]
        floats = U.astype(float)
        flow = np.where(floats > 0, floats * self._h * (1.0 + self._c),
                        floats * self._h * (1.0 - self._c)).sum(axis=1)
        cash_after = C - flow
        volumes = S[None, :] + floats * self._h[None, :]
        wealth_before = S.sum() + C

        F = np.empty((K, self.n_scen))
        Next = np.zeros((K, self.n_scen, self.n_assets + 1))
        for s in range(self.n_scen):
            grown = volumes * np.array(self.stages[stage].coeffs[s])[None, :]
            F[:, s] = grown.sum(axis=1) + cash_after - wealth_before
            if need_next:
                Next[:, s, :-1] = grown
                Next[:, s, -1] = cash_after
        return U, F, Next

    def stage_income(self,
                     state_values: Sequence[float],
                     u: Sequence[int],
                     stage: int,
                     scenario: int) -> float:
        U, F, _ = self.stage_transitions(stage, state_values, need_next=False)
        row = int(np.where((U == np.asarray(u, dtype=np.int64)).all(axis=1))[0][0])
        return float(F[row, scenario])


NEG_INF = float("-inf")


@dataclass(frozen=True)
class PathRecord:

    scenarios: tuple[int, ...]
    prob: float
    states: tuple[tuple[float, ...], ...]
    controls: tuple[tuple[int, ...], ...]
    incomes: tuple[float, ...]
    total_income: float
    final_wealth: float


@dataclass
class Solution:

    criterion_name: str
    root_value: float
    xi1_values: tuple[float, ...]
    paths: list[PathRecord] = field(default_factory=list)
    n_states: dict[int, int] = field(default_factory=dict)
    n_last_stage_solves: int = 0
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

    def __init__(self, model: TransitionModel, xi1_values: Sequence[float] | None = None):
        self.tm = model
        self.xi1 = tuple(float(v) for v in (xi1_values if xi1_values is not None else model.xi1_values))
        self.last = model.n_stages - 1
        self._value: dict[tuple[int, tuple], float] = {}
        self._best_u: dict[tuple[int, tuple], Optional[tuple[int, ...]]] = {}
        self._criterion = None
        self._force_zero = False
        self._t0 = 0.0
        self._heavy_total = 0
        self._heavy_done = 0
        self._last_solves = 0

    def solve(self, criterion, force_zero: bool = False) -> Solution:
        self._value.clear()
        self._best_u.clear()
        self._criterion = criterion
        self._force_zero = force_zero
        self._t0 = time.perf_counter()
        self._heavy_total = 0
        self._heavy_done = 0
        self._last_solves = 0
        root = self._bellman(0, self.xi1)
        if root == NEG_INF:
            raise RuntimeError("Из начального состояния нет допустимых управлений")
        paths = self._forward_pass()
        n_states = dict(Counter(stage for stage, _ in self._value))
        n_states[self.last] = self._last_solves
        return Solution(criterion_name=criterion.name, root_value=root,
                        xi1_values=self.xi1, paths=paths, n_states=n_states,
                        n_last_stage_solves=self._last_solves,
                        elapsed_sec=time.perf_counter() - self._t0)

    def _transitions_for(self, stage: int, state: tuple):
        U, F, Next = self.tm.stage_transitions(stage, state, need_next=stage < self.last)
        if self._force_zero:
            zero_rows = np.where((U == 0).all(axis=1))[0]
            if len(zero_rows):
                U, F, Next = U[zero_rows], F[zero_rows], Next[zero_rows]
        return U, F, Next

    def _bellman(self, stage: int, state: tuple) -> float:
        last = stage == self.last
        memoize = not last
        key = (stage, state)
        if memoize and key in self._value:
            return self._value[key]

        U, F, Next = self._transitions_for(stage, state)
        if U.shape[0] == 0:
            if memoize:
                self._value[key], self._best_u[key] = NEG_INF, None
            return NEG_INF

        if last:
            total = F
            self._last_solves += 1
        else:
            total = np.empty_like(F)
            for s in range(self.tm.n_scen):
                col = Next[:, s, :]
                total[:, s] = F[:, s] + np.array(
                    [self._bellman(stage + 1, tuple(float(x) for x in row)) for row in col])

        scores = self._criterion.scores(total, np.array(self.tm.stages[stage].probs))
        best = int(np.argmax(scores))
        value = float(scores[best])
        if memoize:
            self._value[key] = value
            self._best_u[key] = tuple(int(x) for x in U[best])
        if stage == self.last - 1:
            if self._heavy_total == 0:
                self._heavy_total = max(Next.shape[0] * self.tm.n_scen, 1)
            self._heavy_done += 1
            if self._heavy_done % PROGRESS_EVERY == 0:
                spent = time.perf_counter() - self._t0
                left = spent / self._heavy_done * max(self._heavy_total - self._heavy_done, 0)
                print(f"{self._heavy_done} состояний обработано, "
                      f"прошло {spent / 60:.0f} мин, осталось {left / 60:.0f} мин",
                      flush=True)
        return value

    def _best_control(self, stage: int, state: tuple) -> tuple[int, ...]:
        if stage < self.last:
            return self._best_u[(stage, state)]
        U, F, _ = self._transitions_for(stage, state)
        scores = self._criterion.scores(F, np.array(self.tm.stages[stage].probs))
        return tuple(int(x) for x in U[int(np.argmax(scores))])

    def _forward_pass(self) -> list[PathRecord]:
        paths: list[PathRecord] = []

        def walk(stage: int, state: tuple, scen_acc: tuple[int, ...],
                 prob: float, states: tuple, controls: tuple, incomes: tuple) -> None:
            if stage == self.tm.n_stages:
                total = float(sum(incomes))
                paths.append(PathRecord(scenarios=scen_acc, prob=prob,
                                        states=states, controls=controls,
                                        incomes=incomes, total_income=total,
                                        final_wealth=total + sum(self.xi1)))
                return
            u = self._best_control(stage, state)
            U, F, Next = self.tm.stage_transitions(stage, state, need_next=True)
            row = int(np.where((U == np.asarray(u, dtype=np.int64)).all(axis=1))[0][0])
            for s in range(self.tm.n_scen):
                nxt = tuple(float(x) for x in Next[row, s])
                walk(stage + 1, nxt, scen_acc + (s,),
                     prob * self.tm.stages[stage].probs[s],
                     states + (state,), controls + (u,), incomes + (float(F[row, s]),))

        walk(0, self.xi1, (), 1.0, (), (), ())
        return paths
