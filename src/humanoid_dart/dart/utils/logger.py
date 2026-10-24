"""Lightweight per-iteration metadata summary for the evolutionary pipeline."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

try:
    import wandb as _wandb
except ImportError:  # pragma: no cover
    _wandb = None  # type: ignore[assignment]


@dataclass
class IterationLogger:
    """Collects and reports metadata for a single pipeline iteration."""

    generation: int
    iteration: int

    # Archive bookkeeping
    num_samples: int = 0
    num_passed_fitness: int = 0
    new_bins_filled: int = 0
    old_bins_replaced: int = 0
    bins_rejected: int = 0
    archive_size_before: int = 0
    archive_size_after: int = 0

    # Source breakdown (diffusion_batch, dct_mutation, dct_base, weighted_sum)
    source_counts: Dict[str, int] = field(default_factory=dict)
    elite_source_counts: Dict[str, int] = field(default_factory=dict)

    # Fitness statistics
    fitness_mean: float = 0.0
    fitness_max: float = 0.0
    fitness_min: float = 0.0
    fitness_median: float = 0.0

    # GPU / memory
    cuda_free_gb: float = 0.0
    cuda_total_gb: float = 0.0

    # Section timings (seconds)
    timings: Dict[str, float] = field(default_factory=dict)

    # Wall-clock for the whole iteration
    _iter_start: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def start_timer(self) -> None:
        """Call at the very beginning of an iteration."""
        self._iter_start = time.time()

    def record_timing(self, label: str, elapsed: float) -> None:
        self.timings[label] = round(elapsed, 4)

    def record_sources(self, sources: list[str]) -> None:
        self.source_counts = dict(Counter(sources))

    def record_elite_sources(self, sources: list[str]) -> None:
        self.elite_source_counts = dict(Counter(sources))

    def record_fitness(self, fitness_arr) -> None:
        """Accepts a numpy array of fitness values (one per sample)."""
        if len(fitness_arr) == 0:
            return
        self.fitness_mean = float(fitness_arr.mean())
        self.fitness_max = float(fitness_arr.max())
        self.fitness_min = float(fitness_arr.min())
        import numpy as np
        self.fitness_median = float(np.median(fitness_arr))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Call at the end of the iteration to compute total wall time."""
        if self._iter_start:
            self.timings["total_iteration"] = round(time.time() - self._iter_start, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "iteration": self.iteration,
            "num_samples": self.num_samples,
            "archive": {
                "size_before": self.archive_size_before,
                "size_after": self.archive_size_after,
                "num_passed_fitness": self.num_passed_fitness,
                "new_bins_filled": self.new_bins_filled,
                "old_bins_replaced": self.old_bins_replaced,
                "bins_rejected": self.bins_rejected,
            },
            "fitness": {
                "mean": self.fitness_mean,
                "median": self.fitness_median,
                "max": self.fitness_max,
                "min": self.fitness_min,
            },
            "memory": {
                "cuda_free_gb": self.cuda_free_gb,
                "cuda_total_gb": self.cuda_total_gb,
            },
            "sources": {
                "all_samples": self.source_counts,
                "accepted_elites": self.elite_source_counts,
            },
            "timings": self.timings,
        }

    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def log_wandb(self) -> None:
        if _wandb is None or _wandb.run is None:
            return
        d = self.to_dict()
        flat: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[f"iter/{k}/{sub_k}"] = sub_v
            else:
                flat[f"iter/{k}"] = v
        _wandb.log(flat)

    def print_summary(self) -> None:
        d = self.to_dict()
        width = 60
        print("\n" + "=" * width)
        print(f"  ITERATION SUMMARY  —  Gen {self.generation}  Iter {self.iteration}")
        print("=" * width)
        print(f"  {'Samples generated':<32} {self.num_samples}")

        print("-" * width)
        print("  Archive:")
        for k, v in d["archive"].items():
            print(f"    {k:<28} {v}")

        print("-" * width)
        print("  Fitness:")
        for k, v in d["fitness"].items():
            print(f"    {k:<28} {v:.6f}")

        print("-" * width)
        print("  Memory:")
        for k, v in d["memory"].items():
            print(f"    {k:<28} {v:.2f}")

        if d["sources"]["all_samples"]:
            print("-" * width)
            print("  Sources (all samples):")
            for src, cnt in sorted(d["sources"]["all_samples"].items()):
                print(f"    {src:<28} {cnt}")
        if d["sources"]["accepted_elites"]:
            print("  Sources (accepted elites):")
            for src, cnt in sorted(d["sources"]["accepted_elites"].items()):
                print(f"    {src:<28} {cnt}")

        if self.timings:
            print("-" * width)
            print("  Timings:")
            for label, secs in self.timings.items():
                print(f"    {label:<32} {secs:>8.2f}s")

        print("=" * width + "\n")
