from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import benchmark_functions as bf

from .config import BenchmarkFunctionName


@dataclass(frozen=True, slots=True)
class BenchmarkFunction:
    name: BenchmarkFunctionName
    evaluator: Callable[[list[float]], float]


def schwefel(values: list[float]) -> float:
    return bf.Schwefel(n_dimensions=len(values))(values)


def griewank(values: list[float]) -> float:
    return bf.Griewank(n_dimensions=len(values))(values)


BENCHMARK_FUNCTIONS: dict[BenchmarkFunctionName, BenchmarkFunction] = {
    BenchmarkFunctionName.SCHWEFEL: BenchmarkFunction(BenchmarkFunctionName.SCHWEFEL, schwefel),
    BenchmarkFunctionName.GRIEWANK: BenchmarkFunction(BenchmarkFunctionName.GRIEWANK, griewank),
}
