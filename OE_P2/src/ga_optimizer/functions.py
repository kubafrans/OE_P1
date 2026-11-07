from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import benchmark_functions as bf
import numpy as np
from opfunu.cec_based.cec2014 import F92014

from .config import BenchmarkFunctionName


@dataclass(frozen=True, slots=True)
class BenchmarkFunction:
    name: BenchmarkFunctionName
    bounds: tuple[float, float]
    factory: Callable[[int], object]
    supported_dimensions: tuple[int, ...] | None = None

    def create(self, dimensions: int) -> BenchmarkInstance:
        if self.supported_dimensions and dimensions not in self.supported_dimensions:
            raise ValueError(f"Supported dimensions: {self.supported_dimensions}")
        return BenchmarkInstance(self, dimensions, self.factory(dimensions))


class BenchmarkInstance:
    """One library benchmark per run; support data is loaded only once."""

    def __init__(self, definition: BenchmarkFunction, dimensions: int, function: object) -> None:
        self.definition = definition
        self.dimensions = dimensions
        self.function = function

    def evaluate(self, values: list[float]) -> float:
        if self.definition.name == BenchmarkFunctionName.CEC2014_F9:
            return float(self.function.evaluate(np.asarray(values, dtype=float)))
        return float(self.function(values))

    def minimum(self) -> tuple[float, list[float]]:
        if self.definition.name == BenchmarkFunctionName.CEC2014_F9:
            return float(self.function.f_global), self.function.x_global.tolist()
        minimum = self.function.minimum()
        return float(minimum.score), list(minimum.position)


BENCHMARK_FUNCTIONS: dict[BenchmarkFunctionName, BenchmarkFunction] = {
    BenchmarkFunctionName.SCHWEFEL: BenchmarkFunction(
        BenchmarkFunctionName.SCHWEFEL, (-500.0, 500.0), lambda d: bf.Schwefel(n_dimensions=d)),
    BenchmarkFunctionName.GRIEWANK: BenchmarkFunction(
        BenchmarkFunctionName.GRIEWANK, (-600.0, 600.0), lambda d: bf.Griewank(n_dimensions=d)),
    BenchmarkFunctionName.CEC2014_F9: BenchmarkFunction(
        BenchmarkFunctionName.CEC2014_F9, (-100.0, 100.0), lambda d: F92014(ndim=d),
        (10, 20, 30, 50, 100)),
}
