"""Machine-readable metrics shared by both Flappy Bird training entry points."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import EvaluationSummary


GENERATION_FIELDS = (
    "algorithm",
    "run_seed",
    "generation",
    "population",
    "train_reward_min",
    "train_reward_mean",
    "train_reward_max",
    "validation_reward",
    "best_validation_reward",
    "validation_pipes",
    "validation_steps",
    "training_env_steps",
    "cumulative_training_env_steps",
    "generation_seconds",
)

EPISODE_FIELDS = (
    "algorithm",
    "training_seed",
    "episode",
    "environment_seed",
    "reward",
    "pipes",
    "steps",
    "terminated",
)


def prepare_generation_history(path: str | Path, append: bool = False) -> Path:
    """Create a generation CSV, preserving it only for an explicit resume."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if append and destination.exists() and destination.stat().st_size > 0:
        return destination
    with destination.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=GENERATION_FIELDS).writeheader()
    return destination


def append_generation_metrics(
    path: str | Path,
    values: Mapping[str, Any],
) -> None:
    missing = [field for field in GENERATION_FIELDS if field not in values]
    if missing:
        raise ValueError(f"missing generation metric fields: {', '.join(missing)}")
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATION_FIELDS)
        writer.writerow({field: values[field] for field in GENERATION_FIELDS})


def read_last_generation_metrics(path: str | Path) -> dict[str, str] | None:
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return None
    with source.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        last = None
        for last in rows:
            pass
    return last


def write_episode_results(
    path: str | Path,
    *,
    algorithm: str,
    training_seed: int,
    environment_seeds: Sequence[int],
    summary: EvaluationSummary,
) -> Path:
    """Write one row per held-out episode so distributions remain available."""
    if len(environment_seeds) != len(summary.episodes):
        raise ValueError("environment seed count does not match episode count")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        for index, (seed, episode) in enumerate(
            zip(environment_seeds, summary.episodes),
            start=1,
        ):
            writer.writerow(
                {
                    "algorithm": algorithm,
                    "training_seed": training_seed,
                    "episode": index,
                    "environment_seed": seed,
                    "reward": episode.reward,
                    "pipes": episode.pipes,
                    "steps": episode.steps,
                    "terminated": episode.terminated,
                }
            )
    return destination
