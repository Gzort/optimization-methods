from __future__ import annotations

import os
import sys

from criteria import BayesCriterion
from model import INSTRUMENTS, SCENARIO_SHORT, STAGES, XI1_VALUES
from solver import BellmanSolver, Solution, TransitionModel

W1 = sum(XI1_VALUES)


def build_solver() -> BellmanSolver:
    tm = TransitionModel(INSTRUMENTS, STAGES, XI1_VALUES)
    return BellmanSolver(tm)


_SOLUTION_CACHE: dict[bool, Solution] = {}


def solve_cached(force_zero: bool = False) -> Solution:
    if force_zero not in _SOLUTION_CACHE:
        solver = build_solver()
        _SOLUTION_CACHE[force_zero] = solver.solve(BayesCriterion(), force_zero=force_zero)
    return _SOLUTION_CACHE[force_zero]


def describe_control(u: tuple[int, ...]) -> str:
    parts = []
    for inst, ui in zip(INSTRUMENTS, u):
        if ui > 0:
            parts.append(f"{inst.name}: купить {ui} пак. (+{ui * inst.package:.0f} д.е.)")
        elif ui < 0:
            parts.append(f"{inst.name}: продать {-ui} пак. ({ui * inst.package:.0f} д.е.)")
        else:
            parts.append(f"{inst.name}: без изменений")
    return "; ".join(parts)


def format_solution(sol: Solution) -> str:
    lines = [
        f"Критерий: {sol.criterion_name}",
        f"Начальное состояние: {XI1_VALUES[0]:.0f} / {XI1_VALUES[1]:.0f} / "
        f"{XI1_VALUES[2]:.0f} д.е., свободные средства {XI1_VALUES[3]:.0f} д.е. "
        f"(капитал W1 = {W1:.0f})",
        "",
        f"Значение критерия в корне W1*(Xi_1) = {sol.root_value:.2f}",
        f"Ожидаемый доход E[D] = {sol.expected_income:.2f}; "
        f"худший D = {sol.worst_income:.2f}; лучший D = {sol.best_income:.2f}",
        f"Состояний по слоям: {sol.n_states} (слой -> количество; последний слой решается на лету)",
        f"Время решения: {sol.elapsed_sec:.1f} c",
        "",
        "Оптимальное управление на этапе 1:",
        f"  x1* = {tuple(int(x) for x in sol.paths[0].controls[0])} - "
        f"{describe_control(sol.paths[0].controls[0])}",
        "",
    ]
    tree = format_policy_tree(sol)
    if tree:
        lines.append(tree)
        lines.append("")
    lines.append(format_paths_table(sol))
    return "\n".join(lines)


def format_policy_tree(sol: Solution) -> str:
    n = len(sol.paths[0].controls)
    if n < 2:
        return ""
    lines = ["Дерево оптимальной политики:"]
    for t in range(1, n):
        branches = {}
        for p in sol.paths:
            key = p.scenarios[:t]
            if key not in branches:
                branches[key] = p.controls[t]
        lines.append(f"Этап {t + 1} (управление в зависимости от реализовавшихся ситуаций):")
        for key in sorted(branches):
            tag = "-".join(SCENARIO_SHORT[s] for s in key)
            u = branches[key]
            body = ",".join(f"{x:+d}" for x in u)
            lines.append(f"  после {tag:<6}: x{t + 1}* = ({body})")
    return "\n".join(lines)


def format_paths_table(sol: Solution) -> str:
    n = len(sol.paths[0].controls)
    wcols = " ".join(f"{'w' + str(i + 1):>7}" for i in range(n))
    width = max(20 * n, len("управления по этапам"))
    header = (f"{'№':>2} {'Ситуации':>11} {'P':>6} | {'управления по этапам':>{width}} | "
              f"{wcols} | {'D':>8} {'W4':>8}")
    lines = [f"Политика по траекториям ({len(sol.paths)} комбинаций ситуаций):",
             header, "-" * len(header)]
    for k, p in enumerate(sorted(sol.paths, key=lambda r: -r.prob), start=1):
        scen = "-".join(SCENARIO_SHORT[s] for s in p.scenarios)
        controls = " ; ".join(f"({','.join(f'{x:+d}' for x in u)})" for u in p.controls)
        ws = " ".join(f"{w:>7.1f}" for w in p.incomes)
        lines.append(
            f"{k:>2} {scen:>11} {p.prob:>6.3f} | {controls:>59} | "
            f"{ws} | {p.total_income:>8.1f} {p.final_wealth:>8.1f}")
    return "\n".join(lines)


def save_report(name: str, text: str, out_dir: str = "results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return path


def main() -> None:
    print("Расчёт управления портфелем по критерию Байеса (математическое ожидание дохода)")
    sol = solve_cached()
    text = format_solution(sol)
    print(text)
    save_report("bayes", text)


if __name__ == "__main__":
    main()
