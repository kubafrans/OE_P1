from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import random
from statistics import mean, pstdev
from time import perf_counter
from typing import Callable

from .config import (
    AlgorithmConfig,
    AppConfig,
    ElitismMode,
    ObjectiveMode,
)
from .crossover import crossover
from .encoding import BinaryEncoding
from .functions import BENCHMARK_FUNCTIONS
from .mutation import mutate
from .selection import IndividualView, is_better, order_individuals, select_parents


@dataclass(slots=True)
class GenerationMetrics:
    generation: int
    best_objective: float
    average_objective: float
    std_objective: float


@dataclass(slots=True)
class ChromosomeRecord:
    generation: int
    chromosome: str
    values: list[float]
    objective: float
    fitness: float


@dataclass(slots=True)
class RunMetadata:
    created_at: str
    configuration: dict[str, object]
    objective_mode: str
    benchmark_function: str
    best_objective: float
    best_variables: list[float]
    duration_seconds: float


@dataclass(slots=True)
class RunResult:
    best_individual: IndividualView
    metrics: list[GenerationMetrics]
    population_history: list[ChromosomeRecord]
    duration_seconds: float


class GeneticAlgorithm:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.encoding = BinaryEncoding.build(config.range)
        self.random = random.Random(config.algorithm.seed)
        self.benchmark = BENCHMARK_FUNCTIONS[config.algorithm.benchmark_function]

    def _random_chromosome(self) -> str:
        return "".join(self.random.choice("01") for _ in range(self.encoding.chromosome_length))

    def _evaluate(self, chromosome: str) -> IndividualView:
        values = self.encoding.decode(chromosome)
        objective = self.benchmark.evaluator(values)
        fitness = objective if self.config.algorithm.objective_mode == ObjectiveMode.MAXIMIZE else -objective
        return IndividualView(chromosome=chromosome, values=values, objective=objective, fitness=fitness)

    def _initial_population(self) -> list[IndividualView]:
        return [self._evaluate(self._random_chromosome()) for _ in range(self.config.algorithm.population_size)]

    def _elite_count(self) -> int:
        algorithm = self.config.algorithm
        if algorithm.elite_mode == ElitismMode.COUNT:
            return max(0, min(algorithm.population_size, int(algorithm.elite_value)))
        return max(1, round(algorithm.population_size * algorithm.elite_value / 100.0))

    def _maybe_apply_crossover(self, first: str, second: str) -> tuple[str, str]:
        algorithm = self.config.algorithm
        if self.random.random() <= algorithm.cross_probability:
            return crossover(first, second, algorithm.crossover_method, self.random)
        return first, second

    def _maybe_mutate(self, chromosome: str) -> str:
        algorithm = self.config.algorithm
        if self.random.random() <= algorithm.mutation_probability:
            chromosome = mutate(chromosome, algorithm.mutation_method, self.encoding, self.random)
        if self.random.random() <= algorithm.inversion_probability and len(chromosome) > 1:
            left = self.random.randrange(0, len(chromosome) - 1)
            right = self.random.randrange(left + 1, len(chromosome))
            chromosome = chromosome[:left] + chromosome[left:right][::-1] + chromosome[right:]
        return chromosome

    def _record(self, generation: int, population: list[IndividualView]) -> GenerationMetrics:
        best = order_individuals(population, self.config.algorithm.objective_mode)[0]
        objectives = [item.objective for item in population]
        return GenerationMetrics(
            generation=generation,
            best_objective=best.objective,
            average_objective=mean(objectives),
            std_objective=pstdev(objectives) if len(objectives) > 1 else 0.0,
        )

    def run(
        self,
        progress_callback: Callable[[GenerationMetrics, IndividualView], None] | None = None,
        stop_flag: Callable[[], bool] | None = None,
    ) -> RunResult:
        started = perf_counter()
        population = self._initial_population()
        history: list[GenerationMetrics] = []
        population_history: list[ChromosomeRecord] = []
        best_individual = order_individuals(population, self.config.algorithm.objective_mode)[0]

        for generation in range(self.config.algorithm.epochs):
            if stop_flag and stop_flag():
                break

            metrics = self._record(generation, population)
            history.append(metrics)
            population_history.extend(
                ChromosomeRecord(
                    generation=generation,
                    chromosome=item.chromosome,
                    values=item.values,
                    objective=item.objective,
                    fitness=item.fitness,
                )
                for item in population
            )
            current_best = order_individuals(population, self.config.algorithm.objective_mode)[0]
            if is_better(current_best.objective, best_individual.objective, self.config.algorithm.objective_mode):
                best_individual = current_best
            if progress_callback:
                progress_callback(metrics, best_individual)

            elites = order_individuals(population, self.config.algorithm.objective_mode)[: self._elite_count()]
            selected_count = max(2, round(self.config.algorithm.population_size * self.config.algorithm.selection_best_percent / 100.0))
            mating_pool = select_parents(
                population,
                selected_count,
                self.config.algorithm.selection_method,
                self.config.algorithm.objective_mode,
                self.config.algorithm.tournament_size,
                self.random,
            )

            next_population = [item.chromosome for item in elites]
            while len(next_population) < self.config.algorithm.population_size:
                first_parent = self.random.choice(mating_pool).chromosome
                second_parent = self.random.choice(mating_pool).chromosome
                child_one, child_two = self._maybe_apply_crossover(first_parent, second_parent)
                next_population.append(self._maybe_mutate(child_one))
                if len(next_population) < self.config.algorithm.population_size:
                    next_population.append(self._maybe_mutate(child_two))

            population = [self._evaluate(chromosome) for chromosome in next_population]

        duration = perf_counter() - started
        best_individual = order_individuals(population, self.config.algorithm.objective_mode)[0]
        return RunResult(
            best_individual=best_individual,
            metrics=history,
            population_history=population_history,
            duration_seconds=duration,
        )

    def metadata_for_run(self, result: RunResult) -> RunMetadata:
        return RunMetadata(
            created_at=datetime.now().isoformat(timespec="seconds"),
            configuration={
                "range": asdict(self.config.range),
                "algorithm": {
                    **asdict(self.config.algorithm),
                    "objective_mode": self.config.algorithm.objective_mode.value,
                    "selection_method": self.config.algorithm.selection_method.value,
                    "crossover_method": self.config.algorithm.crossover_method.value,
                    "mutation_method": self.config.algorithm.mutation_method.value,
                    "elite_mode": self.config.algorithm.elite_mode.value,
                    "benchmark_function": self.config.algorithm.benchmark_function.value,
                },
            },
            objective_mode=self.config.algorithm.objective_mode.value,
            benchmark_function=self.config.algorithm.benchmark_function.value,
            best_objective=result.best_individual.objective,
            best_variables=result.best_individual.values,
            duration_seconds=result.duration_seconds,
        )
