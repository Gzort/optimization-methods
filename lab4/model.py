from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:

    name: str
    package: float
    min_volume: float
    commission: float


@dataclass(frozen=True)
class StageParams:

    probs: tuple[float, ...]
    coeffs: tuple[tuple[float, ...], ...]


SCENARIO_NAMES = ("Благоприятная", "Нейтральная", "Негативная")
SCENARIO_SHORT = ("Б", "Н", "Нг")

CB1 = Instrument("ЦБ1", package=25.0, min_volume=30.0, commission=0.04)
CB2 = Instrument("ЦБ2", package=200.0, min_volume=150.0, commission=0.07)
DEP = Instrument("Депозит", package=100.0, min_volume=100.0, commission=0.05)

INSTRUMENTS = (CB1, CB2, DEP)

STAGES = (
    StageParams(probs=(0.60, 0.30, 0.10),
                coeffs=((1.20, 1.10, 1.07),
                        (1.05, 1.02, 1.03),
                        (0.80, 0.95, 1.00))),
    StageParams(probs=(0.30, 0.20, 0.50),
                coeffs=((1.40, 1.15, 1.01),
                        (1.05, 1.00, 1.00),
                        (0.60, 0.90, 1.00))),
    StageParams(probs=(0.40, 0.40, 0.20),
                coeffs=((1.15, 1.12, 1.05),
                        (1.05, 1.01, 1.01),
                        (0.70, 0.94, 1.00))),
)

XI1_VALUES = (100.0, 800.0, 400.0, 600.0)


def validate_problem(instruments, stages, xi1_values):
    if len(xi1_values) != len(instruments) + 1:
        raise ValueError("Состояние должно содержать объёмы всех активов и деньги")
    if any(v < 0 for v in xi1_values):
        raise ValueError("Отрицательные объёмы или деньги недопустимы")
    for inst in instruments:
        if inst.package <= 0:
            raise ValueError(f"Пакет должен быть положительным: {inst.name}")
        if not 0.0 <= inst.commission < 1.0:
            raise ValueError(f"Комиссия должна быть в [0, 1): {inst.name}")
        if inst.min_volume < 0:
            raise ValueError(f"Минимальный объём не может быть отрицательным: {inst.name}")
    for i, stage in enumerate(stages):
        if len(stage.probs) < 1 or len(stage.coeffs) != len(stage.probs):
            raise ValueError(f"Число ситуаций и коэффициентов не согласовано на этапе {i + 1}")
        if any(p < 0 or p > 1 for p in stage.probs):
            raise ValueError(f"Вероятности должны быть в [0, 1] на этапе {i + 1}")
        if abs(sum(stage.probs) - 1.0) > 1e-9:
            raise ValueError(f"Вероятности этапа {i + 1} не суммируются в единицу")
        for row in stage.coeffs:
            if len(row) != len(instruments) or any(k <= 0 for k in row):
                raise ValueError(f"Коэффициенты этапа {i + 1} заданы неверно")
