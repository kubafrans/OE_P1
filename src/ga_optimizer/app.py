from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import (
    AlgorithmConfig,
    AppConfig,
    BenchmarkFunctionName,
    CrossoverMethod,
    ElitismMode,
    MutationMethod,
    ObjectiveMode,
    RangeConfig,
    SelectionMethod,
)
from .engine import GeneticAlgorithm, GenerationMetrics
from .storage import ResultsRepository


@dataclass(slots=True)
class FieldSpec:
    label: str
    default: str


class GeneticApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Genetic Algorithm Optimizer")
        self.geometry("1280x840")
        self.minsize(1120, 760)

        self._running = False
        self._output_path = Path("data") / "results.sqlite3"
        self._exports_root = Path("data") / "exports"
        self.repository = ResultsRepository(self._output_path)
        self._current_config: AppConfig | None = None

        self.range_entries: dict[str, tk.Entry] = {}
        self.algorithm_entries: dict[str, tk.Entry] = {}
        self.option_vars: dict[str, tk.StringVar] = {}
        self._create_layout()

    def _create_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=14)
        left.grid(row=0, column=0, sticky="nsw")
        right = ttk.Frame(self, padding=(0, 14, 14, 14))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_form(left)
        self._build_results(right)

    def _build_form(self, parent: ttk.Frame) -> None:
        title = ttk.Label(parent, text="Konfiguracja", font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        range_frame = ttk.LabelFrame(parent, text="Zakres i reprezentacja", padding=10)
        range_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)

        self._add_entry(range_frame, 0, "begin of range", "-5.12", self.range_entries)
        self._add_entry(range_frame, 1, "end of range", "5.12", self.range_entries)
        self._add_entry(range_frame, 2, "precision", "0.01", self.range_entries)
        self._add_entry(range_frame, 3, "number of parameters", "5", self.range_entries)

        algorithm_frame = ttk.LabelFrame(parent, text="Algorytm", padding=10)
        algorithm_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)

        self._add_entry(algorithm_frame, 0, "population", "40", self.algorithm_entries)
        self._add_entry(algorithm_frame, 1, "epochs", "100", self.algorithm_entries)
        self._add_entry(algorithm_frame, 2, "elite value", "10", self.algorithm_entries)
        self._add_entry(algorithm_frame, 3, "cross probability", "0.9", self.algorithm_entries)
        self._add_entry(algorithm_frame, 4, "mutation probability", "0.05", self.algorithm_entries)
        self._add_entry(algorithm_frame, 5, "inversion probability", "0.02", self.algorithm_entries)
        self._add_entry(algorithm_frame, 6, "percentage best to select", "50", self.algorithm_entries)
        self._add_entry(algorithm_frame, 7, "tournament size", "3", self.algorithm_entries)

        self._add_option(algorithm_frame, 8, "objective mode", [mode.value for mode in ObjectiveMode], ObjectiveMode.MINIMIZE.value)
        self._add_option(algorithm_frame, 9, "elite mode", [mode.value for mode in ElitismMode], ElitismMode.PERCENT.value)
        self._add_option(algorithm_frame, 10, "selection method", [item.value for item in SelectionMethod], SelectionMethod.TOURNAMENT.value)
        self._add_option(algorithm_frame, 11, "cross method", [item.value for item in CrossoverMethod], CrossoverMethod.TWO_POINT.value)
        self._add_option(algorithm_frame, 12, "mutation method", [item.value for item in MutationMethod], MutationMethod.ONE_POINT.value)
        self._add_option(algorithm_frame, 13, "function", [item.value for item in BenchmarkFunctionName], BenchmarkFunctionName.SCHWEFEL.value)

        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 8))
        button_frame.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(button_frame, text="Uruchom", command=self._start_run)
        self.run_button.grid(row=0, column=0, sticky="ew")

        self.status_label = ttk.Label(parent, text="Gotowe do uruchomienia.")
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _add_entry(self, parent: ttk.Frame, row: int, label: str, default: str, store: dict[str, tk.Entry]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, width=18)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)
        parent.columnconfigure(1, weight=1)
        store[label] = entry

    def _add_option(self, parent: ttk.Frame, row: int, label: str, options: list[str], default: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        value = tk.StringVar(value=default)
        box = ttk.Combobox(parent, values=options, state="readonly", textvariable=value)
        box.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)
        self.option_vars[label] = value

    def _build_results(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)

        self.duration_var = tk.StringVar(value="Czas: -")
        self.best_var = tk.StringVar(value="Najlepszy wynik: -")
        self.variables_var = tk.StringVar(value="Najlepsze zmienne: -")
        ttk.Label(top, textvariable=self.duration_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.best_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w")
        ttk.Label(top, textvariable=self.variables_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="w")

        content = ttk.Notebook(parent)
        content.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        parent.rowconfigure(1, weight=1)

        best_tab = ttk.Frame(content)
        average_tab = ttk.Frame(content)
        std_tab = ttk.Frame(content)
        log_tab = ttk.Frame(content)
        content.add(best_tab, text="Best")
        content.add(average_tab, text="Average")
        content.add(std_tab, text="Std")
        content.add(log_tab, text="Log")

        for tab in (best_tab, average_tab, std_tab):
            tab.rowconfigure(0, weight=1)
            tab.columnconfigure(0, weight=1)

        self.figure_best = Figure(figsize=(7.5, 3.6), dpi=100)
        self.axis_best = self.figure_best.add_subplot(111)
        self.figure_average = Figure(figsize=(7.5, 3.6), dpi=100)
        self.axis_average = self.figure_average.add_subplot(111)
        self.figure_std = Figure(figsize=(7.5, 3.6), dpi=100)
        self.axis_std = self.figure_std.add_subplot(111)

        self.best_canvas = FigureCanvasTkAgg(self.figure_best, master=best_tab)
        self.best_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.average_canvas = FigureCanvasTkAgg(self.figure_average, master=average_tab)
        self.average_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.std_canvas = FigureCanvasTkAgg(self.figure_std, master=std_tab)
        self.std_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.log_text = tk.Text(log_tab, wrap="word", height=18)
        self.log_text.pack(fill="both", expand=True)

    def _read_float(self, entries: dict[str, tk.Entry], key: str) -> float:
        return float(entries[key].get().strip())

    def _read_int(self, entries: dict[str, tk.Entry], key: str) -> int:
        return int(float(entries[key].get().strip()))

    def _build_config(self) -> AppConfig:
        range_config = RangeConfig(
            lower=self._read_float(self.range_entries, "begin of range"),
            upper=self._read_float(self.range_entries, "end of range"),
            precision=self._read_float(self.range_entries, "precision"),
            dimensions=self._read_int(self.range_entries, "number of parameters"),
        )
        algorithm = AlgorithmConfig(
            population_size=self._read_int(self.algorithm_entries, "population"),
            epochs=self._read_int(self.algorithm_entries, "epochs"),
            elite_mode=ElitismMode(self.option_vars["elite mode"].get()),
            elite_value=float(self.algorithm_entries["elite value"].get().strip()),
            cross_probability=float(self.algorithm_entries["cross probability"].get().strip()),
            mutation_probability=float(self.algorithm_entries["mutation probability"].get().strip()),
            inversion_probability=float(self.algorithm_entries["inversion probability"].get().strip()),
            selection_method=SelectionMethod(self.option_vars["selection method"].get()),
            selection_best_percent=float(self.algorithm_entries["percentage best to select"].get().strip()),
            tournament_size=self._read_int(self.algorithm_entries, "tournament size"),
            crossover_method=CrossoverMethod(self.option_vars["cross method"].get()),
            mutation_method=MutationMethod(self.option_vars["mutation method"].get()),
            objective_mode=ObjectiveMode(self.option_vars["objective mode"].get()),
            benchmark_function=BenchmarkFunctionName(self.option_vars["function"].get()),
        )
        return AppConfig(range=range_config, algorithm=algorithm)

    def _start_run(self) -> None:
        if self._running:
            return
        try:
            config = self._build_config()
        except ValueError as error:
            messagebox.showerror("Błąd konfiguracji", str(error))
            return

        if config.range.upper <= config.range.lower:
            messagebox.showerror("Błąd konfiguracji", "End of range must be greater than begin of range.")
            return
        if config.range.precision <= 0:
            messagebox.showerror("Błąd konfiguracji", "Precision must be greater than zero.")
            return

        self._current_config = config
        self._running = True
        self.run_button.configure(state="disabled")
        self.status_label.configure(text="Trwa obliczanie...")
        self._append_log("Start obliczeń")

        def worker() -> None:
            engine = GeneticAlgorithm(config)

            def progress(metrics: GenerationMetrics, best) -> None:
                self.after(0, lambda: self._update_progress(metrics, best))

            result = engine.run(progress_callback=progress)
            run_id = self.repository.save_run(engine.metadata_for_run(result), result.metrics)
            self.after(0, lambda: self._finish_run(result, run_id))

        threading.Thread(target=worker, daemon=True).start()

    def _update_progress(self, metrics: GenerationMetrics, best) -> None:
        self.status_label.configure(text=f"Epoka {metrics.generation + 1}: najlepszy wynik {best.objective:.6f}")
        self._append_log(
            f"Epoka {metrics.generation + 1}: best={metrics.best_objective:.6f}, avg={metrics.average_objective:.6f}, std={metrics.std_objective:.6f}"
        )

    def _finish_run(self, result, run_id: int) -> None:
        self._running = False
        self.run_button.configure(state="normal")
        self.duration_var.set(f"Czas: {result.duration_seconds:.3f}s")
        self.best_var.set(f"Najlepszy wynik: {result.best_individual.objective:.6f} (run #{run_id})")
        self.variables_var.set(
            "Najlepsze zmienne: " + ", ".join(f"{value:.4f}" for value in result.best_individual.values)
        )
        self.status_label.configure(text="Obliczenia zakończone.")
        self._draw_plots(result.metrics)
        self._export_artifacts(run_id, result)
        self._append_log(f"Zapisano wynik do bazy: {self._output_path}")

    def _draw_plots(self, metrics: list[GenerationMetrics]) -> None:
        generations = [record.generation + 1 for record in metrics]
        best_values = [record.best_objective for record in metrics]
        average_values = [record.average_objective for record in metrics]
        std_values = [record.std_objective for record in metrics]

        self.axis_best.clear()
        self.axis_best.set_title("Wartości funkcji od kolejnej iteracji")
        self.axis_best.set_xlabel("Iteracja")
        self.axis_best.set_ylabel("Best")
        self.axis_best.plot(generations, best_values, color="#0f766e", linewidth=2)
        self.axis_best.grid(True, alpha=0.25)
        self.figure_best.tight_layout()
        self.best_canvas.draw_idle()

        self.axis_average.clear()
        self.axis_average.set_title("Średnia wartości funkcji")
        self.axis_average.set_xlabel("Iteracja")
        self.axis_average.set_ylabel("Średnia")
        self.axis_average.plot(generations, average_values, color="#1d4ed8", linewidth=2)
        self.axis_average.grid(True, alpha=0.25)
        self.figure_average.tight_layout()
        self.average_canvas.draw_idle()

        self.axis_std.clear()
        self.axis_std.set_title("Odchylenie standardowe wartości funkcji")
        self.axis_std.set_xlabel("Iteracja")
        self.axis_std.set_ylabel("Odchylenie")
        self.axis_std.plot(generations, std_values, color="#b45309", linewidth=2)
        self.axis_std.grid(True, alpha=0.25)
        self.figure_std.tight_layout()
        self.std_canvas.draw_idle()

    def _export_artifacts(self, run_id: int, result) -> None:
        if self._current_config is None:
            return
        export_dir = self._exports_root / f"run_{run_id}"
        export_dir.mkdir(parents=True, exist_ok=True)
        self.repository.export_population_csv(
            export_dir / "iterations.csv",
            result.metrics,
            self._current_config.algorithm.objective_mode,
        )
        self.figure_best.savefig(export_dir / "best_value.png", dpi=150, bbox_inches="tight")
        self.figure_average.savefig(export_dir / "average_value.png", dpi=150, bbox_inches="tight")
        self.figure_std.savefig(export_dir / "std_value.png", dpi=150, bbox_inches="tight")
        self._append_log(f"Wyeksportowano CSV i obrazy do: {export_dir}")

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")


def main() -> None:
    app = GeneticApp()
    app.mainloop()
