from __future__ import annotations

import argparse
import os


from criteria import (BayesCriterion, HurwiczCriterion, LaplaceCriterion,
                      SavageCriterion, WaldCriterion)
from model import CASH_GRID, INSTRUMENTS, SCENARIO_SHORT, STAGES, XI1_VALUES
from solver import BellmanSolver, Solution, TransitionModel

W1 = sum(XI1_VALUES)

CRITERIA = {
    "bayes": BayesCriterion,
    "laplace": LaplaceCriterion,
    "wald": WaldCriterion,
    "hurwicz": HurwiczCriterion,
    "savage": SavageCriterion,
}

def build_solver() -> BellmanSolver:
    tm = TransitionModel(INSTRUMENTS, STAGES, CASH_GRID)
    return BellmanSolver(tm, XI1_VALUES)

_SOLUTION_CACHE: dict[tuple[str, float], Solution] = {}

def solve_cached(name: str, alpha: float = 0.5, force_zero: bool = False) -> Solution:
    key = (name + ("#zero" if force_zero else ""), alpha)
    if key not in _SOLUTION_CACHE:
        cls = CRITERIA[name]
        criterion = cls(alpha) if cls is HurwiczCriterion else cls()
        solver = build_solver()
        _SOLUTION_CACHE[key] = solver.solve(criterion, force_zero=force_zero)
    return _SOLUTION_CACHE[key]

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
        f"Состояний в таблицах Беллмана: {sol.n_states} (этап -> количество)",
        f"Время решения: {sol.elapsed_sec:.1f} c",
        "",
        "Оптимальное управление на этапе 1:",
        f"  x1* = {tuple(int(x) for x in sol.paths[0].controls[0])} - "
        f"{describe_control(sol.paths[0].controls[0])}",
        "",
    ]
    lines.append(format_paths_table(sol))
    return "\n".join(lines)

def format_paths_table(sol: Solution) -> str:
    header = (f"{'№':>2} {'Ситуации':>11} {'P':>6} | "
              f"{'x1* (пакеты ЦБ1,ЦБ2,Деп)':>26} {'x2*':>15} {'x3*':>15} | "
              f"{'w1':>7} {'w2':>7} {'w3':>7} | {'D':>8} {'W4':>8}")
    lines = ["Политика по траекториям (27 комбинаций ситуаций):", header, "-" * len(header)]
    for k, p in enumerate(sorted(sol.paths, key=lambda r: -r.prob), start=1):
        scen = "-".join(SCENARIO_SHORT[s] for s in p.scenarios)
        controls = " ; ".join(f"({','.join(f'{x:+d}' for x in u)})" for u in p.controls)
        lines.append(
            f"{k:>2} {scen:>11} {p.prob:>6.3f} | {controls:>59} | "
            f"{p.incomes[0]:>7.1f} {p.incomes[1]:>7.1f} {p.incomes[2]:>7.1f} | "
            f"{p.total_income:>8.1f} {p.final_wealth:>8.1f}")
    return "\n".join(lines)

def run_comparison(alphas: tuple[float, ...] = (0.5,)) -> str:
    rows = []
    entries = [("bayes", 0.5, False), ("laplace", 0.5, False), ("wald", 0.5, False)]
    entries += [("hurwicz", a, False) for a in alphas]
    entries += [("savage", 0.5, False), ("bayes", 0.5, True)]
    for name, alpha, zero in entries:
        sol = solve_cached(name, alpha, force_zero=zero)
        label = sol.criterion_name + (" [базовая: держать]" if zero else "")
        rows.append((label, sol.root_value, sol.expected_income,
                     sol.worst_income, sol.best_income))

    header = (f"{'Критерий':<28} {'Значение':>9} {'E[D]':>8} {'худший D':>9} "
              f"{'лучший D':>9}")
    lines = ["Сравнение критериев принятия решений:", header, "-" * len(header)]
    for label, value, mean, worst, best in rows:
        lines.append(f"{label:<28} {value:>9.2f} {mean:>8.2f} {worst:>9.2f} {best:>9.2f}")
    return "\n".join(lines)

def save_report(name: str, text: str, out_dir: str = "results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return path

MENU = """\
Выберите критерий принятия решений:
  1 - Байес (математическое ожидание)
  2 - Лаплас (равновероятные ситуации)
  3 - Вальд (максимин)
  4 - Гурвиц (коэффициент оптимизма, по умолчанию alpha = 0.5)
  5 - Сэвидж (минимаксное сожаление)
  6 - сравнение всех критериев
Ваш выбор (1-6): """

def interactive() -> None:
    choice = input(MENU).strip()
    names = {"1": "bayes", "2": "laplace", "3": "wald", "4": "hurwicz", "5": "savage"}
    if choice in names:
        alpha = 0.5
        if choice == "4":
            alpha = float(input("Коэффициент оптимизма alpha в [0,1]: ").strip() or "0.5")
        sol = solve_cached(names[choice], alpha)
        text = format_solution(sol)
        print(text)
        save_report(names[choice], text)
    elif choice == "6":
        text = run_comparison()
        print(text)
        save_report("comparison", text)
    else:
        print("Неизвестный выбор; запустите с --criterion или --compare.")

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ЛР4: ДП, управление портфелем (Беллман)")
    parser.add_argument("--criterion", choices=sorted(CRITERIA),
                        action="append",
                        help="критерий (можно несколько): bayes | hurwicz | laplace | savage | wald")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="коэффициент оптимизма для критерия Гурвица (по умолчанию 0.5)")
    parser.add_argument("--compare", action="store_true",
                        help="решить по всем критериям и вывести сравнение")
    args = parser.parse_args(argv)

    if args.compare or args.criterion:
        if args.compare:
            text = run_comparison(alphas=(args.alpha,))
            print(text)
            save_report("comparison", text)
        for name in args.criterion or []:
            sol = solve_cached(name, args.alpha)
            text = format_solution(sol)
            print(text)
            save_report(name, text)
    else:
        interactive()

if __name__ == "__main__":
    main()
