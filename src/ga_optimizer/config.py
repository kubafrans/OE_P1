from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ObjectiveMode(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SelectionMethod(str, Enum):
    BEST = "best"
    WORST = "worst"
    ROULETTE = "roulette"
    TOURNAMENT = "tournament"


class CrossoverMethod(str, Enum):
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    GRANULAR = "granular"


class MutationMethod(str, Enum):
    BOUNDARY = "boundary"
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"


class ElitismMode(str, Enum):
    PERCENT = "percent"
    COUNT = "count"


class BenchmarkFunctionName(str, Enum):
    SCHWEFEL = "Schwefel"
    GRIEWANK = "Griewank"
    CEC2014_F9 = "CEC 2014 F9 - Shifted and Rotated Rastrigin"


@dataclass(slots=True)
class RangeConfig:
    lower: float = -500.0
    upper: float = 500.0
    precision: float = 0.01
    dimensions: int = 5


@dataclass(slots=True)
class AlgorithmConfig:
    population_size: int = 40
    epochs: int = 100
    elite_mode: ElitismMode = ElitismMode.PERCENT
    elite_value: float = 10.0
    cross_probability: float = 0.9
    mutation_probability: float = 0.05
    inversion_probability: float = 0.02
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    selection_best_percent: float = 50.0
    tournament_size: int = 3
    crossover_method: CrossoverMethod = CrossoverMethod.TWO_POINT
    mutation_method: MutationMethod = MutationMethod.ONE_POINT
    objective_mode: ObjectiveMode = ObjectiveMode.MINIMIZE
    benchmark_function: BenchmarkFunctionName = BenchmarkFunctionName.SCHWEFEL
    seed: int | None = None


@dataclass(slots=True)
class AppConfig:
    range: RangeConfig
    algorithm: AlgorithmConfig

    def validate(self) -> None:
        r, a = self.range, self.algorithm
        if not all(isfinite(value) for value in (r.lower, r.upper, r.precision)):
            raise ValueError("Bounds and precision must be finite numbers.")
        if r.upper <= r.lower or r.precision <= 0:
            raise ValueError("Upper bound must exceed lower bound; precision must be positive.")
        for label, value, minimum in (
            ("Dimensions", r.dimensions, 1), ("Population", a.population_size, 2),
            ("Epochs", a.epochs, 1), ("Tournament size", a.tournament_size, 2),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}.")
        for label, value in (("Crossover", a.cross_probability), ("Mutation", a.mutation_probability),
                             ("Inversion", a.inversion_probability)):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} probability must be between 0 and 1.")
        if not isfinite(a.selection_best_percent) or not 0 < a.selection_best_percent <= 100:
            raise ValueError("Selection percentage must be in (0, 100].")
        limit = 100 if a.elite_mode == ElitismMode.PERCENT else a.population_size
        if not isfinite(a.elite_value) or not 0 <= a.elite_value <= limit:
            raise ValueError(f"Elite value must be between 0 and {limit}.")
        if a.elite_mode == ElitismMode.COUNT and a.elite_value != int(a.elite_value):
            raise ValueError("Elite count must be an integer.")
        if a.benchmark_function == BenchmarkFunctionName.CEC2014_F9:
            if r.dimensions not in (10, 20, 30, 50, 100):
                raise ValueError("CEC 2014 F9 supports 10, 20, 30, 50 or 100 dimensions.")
            if r.lower != -100 or r.upper != 100:
                raise ValueError("CEC 2014 F9 requires bounds [-100, 100].")
