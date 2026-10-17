from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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


@dataclass(slots=True)
class RangeConfig:
    lower: float = -5.12
    upper: float = 5.12
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
