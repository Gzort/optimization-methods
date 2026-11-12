from dataclasses import dataclass

@dataclass(frozen=True)
class Instrument:

    name: str
    package: float
    min_volume: float
    commission: float
    grid: float

@dataclass(frozen=True)
class StageParams:

    probs: tuple[float, float, float]
    coeffs: tuple[tuple[float, float, float], ...]

SCENARIO_NAMES = ("Благоприятная", "Нейтральная", "Негативная")
SCENARIO_SHORT = ("Б", "Н", "Нг")

CB1 = Instrument("ЦБ1", package=25.0, min_volume=30.0, commission=0.04, grid=25.0)
CB2 = Instrument("ЦБ2", package=200.0, min_volume=150.0, commission=0.07, grid=50.0)
DEP = Instrument("Депозит", package=100.0, min_volume=100.0, commission=0.05, grid=25.0)

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

CASH_GRID = 25.0
