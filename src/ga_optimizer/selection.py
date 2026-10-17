from __future__ import annotations

import random
from dataclasses import dataclass

from .config import ObjectiveMode, SelectionMethod


@dataclass(slots=True)
class IndividualView:
    chromosome: str
    values: list[float]
    objective: float
    fitness: float


def is_better(left: float, right: float, objective_mode: ObjectiveMode) -> bool:
    if objective_mode == ObjectiveMode.MAXIMIZE:
        return left > right
    return left < right


def order_individuals(individuals: list[IndividualView], objective_mode: ObjectiveMode) -> list[IndividualView]:
    return sorted(individuals, key=lambda item: item.objective, reverse=objective_mode == ObjectiveMode.MAXIMIZE)


def selection_weights(individuals: list[IndividualView], objective_mode: ObjectiveMode) -> list[float]:
    if not individuals:
        return []
    objectives = [item.objective for item in individuals]
    if objective_mode == ObjectiveMode.MAXIMIZE:
        baseline = min(objectives)
        return [value - baseline + 1e-9 for value in objectives]
    baseline = max(objectives)
    return [baseline - value + 1e-9 for value in objectives]


def select_parents(
    individuals: list[IndividualView],
    count: int,
    method: SelectionMethod,
    objective_mode: ObjectiveMode,
    tournament_size: int,
    rng: random.Random,
) -> list[IndividualView]:
    if not individuals:
        return []
    count = max(2, count)
    if method == SelectionMethod.BEST:
        return order_individuals(individuals, objective_mode)[:count]
    if method == SelectionMethod.WORST:
        return list(reversed(order_individuals(individuals, objective_mode)))[:count]
    if method == SelectionMethod.ROULETTE:
        weights = selection_weights(individuals, objective_mode)
        return [rng.choices(individuals, weights=weights, k=1)[0] for _ in range(count)]
    selected: list[IndividualView] = []
    tournament_size = max(2, min(tournament_size, len(individuals)))
    for _ in range(count):
        candidates = rng.sample(individuals, k=tournament_size)
        selected.append(order_individuals(candidates, objective_mode)[0])
    return selected
