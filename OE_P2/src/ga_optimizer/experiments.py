from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .config import AppConfig, ObjectiveMode
from .engine import GeneticAlgorithm, RunMetadata, RunResult
from .storage import ResultsRepository


@dataclass(slots=True)
class ExperimentResult:
    summary: dict[str, object]
    best_run: RunResult
    best_run_id: int
    export_directory: Path


def export_run(directory: Path, result: RunResult, metadata: RunMetadata,
               repository: ResultsRepository) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    repository.export_population_csv(directory / "iterations.csv", result.metrics,
                                     ObjectiveMode(metadata.objective_mode))
    (directory / "run.json").write_text(json.dumps(asdict(metadata), indent=2, ensure_ascii=False), encoding="utf-8")


def export_plots(directory: Path, result: RunResult) -> None:
    generations = [item.generation for item in result.metrics]
    for attribute, filename, label in (
        ("best_objective", "best_value.png", "Najlepsza wartość w populacji"),
        ("average_objective", "average_value.png", "Średnia wartość w populacji"),
        ("std_objective", "std_value.png", "Odchylenie standardowe w populacji"),
    ):
        figure = Figure(figsize=(8, 4), dpi=120)
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        axis.plot(generations, [getattr(item, attribute) for item in result.metrics])
        axis.set(xlabel="Generacja (0 = populacja początkowa)", ylabel=label)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(directory / filename)


class ExperimentRunner:
    """Repeat a configuration with distinct seeds and retain every iteration."""

    def __init__(self, repository: ResultsRepository, exports_root: Path) -> None:
        self.repository = repository
        self.exports_root = exports_root

    def run(self, config: AppConfig, repetitions: int = 10, base_seed: int = 42,
            progress: Callable[[int, int], None] | None = None) -> ExperimentResult:
        config.validate()
        if repetitions < 1:
            raise ValueError("Repetitions must be positive.")
        rows: list[dict[str, object]] = []
        best_run: RunResult | None = None
        best_run_id = 0
        first_run_id = 0
        minimize = config.algorithm.objective_mode == ObjectiveMode.MINIMIZE
        for index in range(repetitions):
            run_config = replace(config, algorithm=replace(config.algorithm, seed=base_seed + index))
            engine = GeneticAlgorithm(run_config)
            result = engine.run()
            metadata = engine.metadata_for_run(result)
            run_id = self.repository.save_run(metadata, result.metrics)
            if index == 0:
                first_run_id = run_id
            export_run(self.exports_root / f"run_{run_id}", result, metadata, self.repository)
            rows.append({"run_id": run_id, "seed": base_seed + index,
                         "best_objective": result.best_individual.objective,
                         "duration_seconds": result.duration_seconds})
            if best_run is None or (result.best_individual.objective < best_run.best_individual.objective
                                    if minimize else result.best_individual.objective > best_run.best_individual.objective):
                best_run, best_run_id = result, run_id
            if progress:
                progress(index + 1, repetitions)
        assert best_run is not None
        scores = [float(row["best_objective"]) for row in rows]
        times = [float(row["duration_seconds"]) for row in rows]
        minimum, arguments = engine.benchmark.minimum()
        directory = self.exports_root / f"experiment_{first_run_id}"
        directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "function": config.algorithm.benchmark_function.value,
            "dimensions": config.range.dimensions, "objective_mode": config.algorithm.objective_mode.value,
            "repetitions": repetitions, "base_seed": base_seed,
            "mean": mean(scores), "std_between_runs": pstdev(scores),
            "best": min(scores) if minimize else max(scores),
            "worst": max(scores) if minimize else min(scores),
            "mean_duration_seconds": mean(times), "best_run_id": best_run_id,
            "reference_minimum": minimum, "reference_arguments": arguments,
            "absolute_error": abs(best_run.best_individual.objective - minimum) if minimize else None,
            "best_variables": best_run.best_individual.values,
            "configuration": {**metadata.configuration, "algorithm": {
                **metadata.configuration["algorithm"], "seed": base_seed}},
        }
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        with (directory / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        best_directory = self.exports_root / f"run_{best_run_id}"
        export_plots(best_directory, best_run)
        return ExperimentResult(summary, best_run, best_run_id, directory)
