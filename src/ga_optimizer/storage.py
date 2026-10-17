from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import ObjectiveMode
from .engine import GenerationMetrics, RunMetadata


class ResultsRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    configuration TEXT NOT NULL,
                    objective_mode TEXT NOT NULL,
                    benchmark_function TEXT NOT NULL,
                    best_objective REAL NOT NULL,
                    best_variables TEXT NOT NULL,
                    duration_seconds REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    generation INTEGER NOT NULL,
                    best_objective REAL NOT NULL,
                    average_objective REAL NOT NULL,
                    std_objective REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )

    def save_run(self, metadata: RunMetadata, metrics: list[GenerationMetrics]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (created_at, configuration, objective_mode, benchmark_function, best_objective, best_variables, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.created_at,
                    json.dumps(metadata.configuration, ensure_ascii=False),
                    metadata.objective_mode,
                    metadata.benchmark_function,
                    metadata.best_objective,
                    json.dumps(metadata.best_variables),
                    metadata.duration_seconds,
                ),
            )
            run_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO generation_metrics (run_id, generation, best_objective, average_objective, std_objective)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        record.generation,
                        record.best_objective,
                        record.average_objective,
                        record.std_objective,
                    )
                    for record in metrics
                ],
            )
            return int(run_id)

    def export_population_csv(
        self,
        export_path: Path,
        records: list[GenerationMetrics],
        objective_mode: ObjectiveMode,
    ) -> None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = ["iteration", "best", "mean", "fitness", "std"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                fitness = record.best_objective if objective_mode == ObjectiveMode.MAXIMIZE else -record.best_objective
                writer.writerow(
                    {
                        "iteration": record.generation,
                        "best": record.best_objective,
                        "mean": record.average_objective,
                        "fitness": fitness,
                        "std": record.std_objective,
                    }
                )
