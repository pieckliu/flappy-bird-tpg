"""Run a reproducible Standard-TPG vs Weighted-TPG Flappy Bird experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from statistics import fmean, median, stdev
from typing import Any

import numpy as np

from shared_flappy import FlappyEnv, evaluate_agent, make_seeds
from shared_flappy.metrics import (
    EPISODE_FIELDS,
    GENERATION_FIELDS,
    append_generation_metrics,
    prepare_generation_history,
    write_episode_results,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ALGORITHMS = ("weighted_tpg", "standard_tpg")
DISPLAY_NAMES = {
    "weighted_tpg": "Weighted TPG",
    "standard_tpg": "Standard TPG (PyTPG)",
}


def default_pytpg_python() -> Path:
    windows = PROJECT_ROOT / "pytpg_flappy" / ".venv" / "Scripts" / "python.exe"
    posix = PROJECT_ROOT / "pytpg_flappy" / ".venv" / "bin" / "python"
    return windows if windows.exists() else posix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=100_000)
    parser.add_argument("--test-seed", type=int, default=100_000_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/tpg_comparison"),
    )
    parser.add_argument(
        "--weighted-python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--pytpg-python",
        type=Path,
        default=default_pytpg_python(),
    )
    parser.add_argument(
        "--plot-x",
        choices=("env-steps", "generation"),
        default="env-steps",
        help="learning-curve x axis; environment steps is the fairer default",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete and rerun only the seed directories selected by this command",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="rebuild combined CSV files and the figure from existing runs",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue other seeds if one worker process fails",
    )

    # Internal worker mode. The coordinator invokes this script with the
    # appropriate isolated Python interpreter.
    parser.add_argument(
        "--worker",
        choices=ALGORITHMS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--run-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_positive(args: argparse.Namespace) -> None:
    for name in (
        "runs",
        "generations",
        "population",
        "episodes",
        "validation_episodes",
        "test_episodes",
        "max_steps",
        "seed_stride",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")


def training_seed(args: argparse.Namespace, run_index: int) -> int:
    return args.base_seed + run_index * args.seed_stride


def run_directory(output: Path, algorithm: str, seed: int) -> Path:
    return output / algorithm / f"seed_{seed:010d}"


def print_progress(
    *,
    algorithm: str,
    generation: int,
    total_generations: int,
    rewards: list[float],
    validation: Any,
    best_validation: float,
    population: int,
    seconds: float,
) -> None:
    print(
        f"algorithm={algorithm} "
        f"generation={generation:04d} "
        f"iteration={generation:04d}/{total_generations:04d} "
        f"agents={population:04d} "
        f"min={min(rewards):8.2f} "
        f"avg={fmean(rewards):8.2f} "
        f"max={max(rewards):8.2f} "
        f"val={validation.reward:8.2f} "
        f"best={best_validation:8.2f} "
        f"val_pipes={validation.pipes:7.2f} "
        f"val_steps={validation.steps:8.2f} "
        f"seconds={seconds:7.2f}",
        flush=True,
    )


def run_weighted_worker(args: argparse.Namespace) -> None:
    from tpg import TPGAgent, evolve

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = prepare_generation_history(run_dir / "history.csv")
    checkpoint = run_dir / "best.json"
    training_seeds = make_seeds(args.run_seed, args.episodes)
    validation_seeds = make_seeds(
        args.run_seed,
        args.validation_episodes,
        validation=True,
    )

    best_validation = float("-inf")
    generation_started = time.perf_counter()
    generation_training_steps = 0
    cumulative_training_steps = 0

    def evaluate(agent: Any) -> float:
        nonlocal generation_training_steps, cumulative_training_steps
        summary = evaluate_agent(agent, training_seeds, args.max_steps)
        steps = sum(episode.steps for episode in summary.episodes)
        generation_training_steps += steps
        cumulative_training_steps += steps
        return summary.reward

    def record_generation(
        generation: int,
        scored: list[tuple[float, Any]],
        _training_best: float,
        _best_agent: Any,
    ) -> None:
        nonlocal best_validation, generation_started, generation_training_steps
        current_score, current_agent = scored[0]
        validation = evaluate_agent(current_agent, validation_seeds, args.max_steps)
        if validation.reward > best_validation:
            best_validation = validation.reward
            current_agent.save(checkpoint)

        rewards = [score for score, _ in scored]
        elapsed = time.perf_counter() - generation_started
        print_progress(
            algorithm="weighted_tpg",
            generation=generation + 1,
            total_generations=args.generations,
            rewards=rewards,
            validation=validation,
            best_validation=best_validation,
            population=len(scored),
            seconds=elapsed,
        )
        append_generation_metrics(
            history_path,
            {
                "algorithm": "weighted_tpg",
                "run_seed": args.run_seed,
                "generation": generation + 1,
                "population": len(scored),
                "train_reward_min": min(rewards),
                "train_reward_mean": fmean(rewards),
                "train_reward_max": current_score,
                "validation_reward": validation.reward,
                "best_validation_reward": best_validation,
                "validation_pipes": validation.pipes,
                "validation_steps": validation.steps,
                "training_env_steps": generation_training_steps,
                "cumulative_training_env_steps": cumulative_training_steps,
                "generation_seconds": elapsed,
            },
        )
        generation_training_steps = 0
        generation_started = time.perf_counter()

    evolve(
        evaluate,
        generations=args.generations,
        population_size=args.population,
        seed=args.run_seed,
        on_scored=record_generation,
    )

    agent = TPGAgent.load(checkpoint)
    test_seeds = make_seeds(args.test_seed, args.test_episodes)
    test_summary = evaluate_agent(agent, test_seeds, args.max_steps)
    write_episode_results(
        run_dir / "test_episodes.csv",
        algorithm="weighted_tpg",
        training_seed=args.run_seed,
        environment_seeds=test_seeds,
        summary=test_summary,
    )
    print(
        f"test algorithm=weighted_tpg seed={args.run_seed} "
        f"reward={test_summary.reward:.3f} pipes={test_summary.pipes:.3f} "
        f"steps={test_summary.steps:.3f}",
        flush=True,
    )


def create_standard_trainer(Trainer: Any, population: int) -> Any:
    return Trainer(
        actions=FlappyEnv.action_count,
        inputSize=FlappyEnv.observation_size,
        teamPopSize=population,
        rootBasedPop=True,
        gap=0.5,
        nRegisters=4,
        initMaxTeamSize=4,
        initMaxProgSize=16,
        maxTeamSize=8,
        pLrnDel=0.5,
        pLrnAdd=0.5,
        pLrnMut=0.3,
        pProgMut=0.66,
        pActMut=0.33,
        pActAtom=0.7,
        pInstDel=0.4,
        pInstAdd=0.4,
        pInstSwp=0.5,
        pInstMut=0.8,
        doElites=True,
        memType=None,
        rampancy=(0, 0, 0),
        operationSet="def",
        traversal="team",
    )


def run_standard_worker(args: argparse.Namespace) -> None:
    pytpg_directory = PROJECT_ROOT / "pytpg_flappy"
    if str(pytpg_directory) not in sys.path:
        sys.path.insert(0, str(pytpg_directory))
    from pytpg_loader import load_pytpg

    task_name = "flappy-bird-shared"
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = prepare_generation_history(run_dir / "history.csv")
    checkpoint = run_dir / "best_agent.pkl"

    random.seed(args.run_seed)
    np.random.seed(args.run_seed)
    Trainer, Agent = load_pytpg()
    trainer = create_standard_trainer(Trainer, args.population)
    training_seeds = make_seeds(args.run_seed, args.episodes)
    validation_seeds = make_seeds(
        args.run_seed,
        args.validation_episodes,
        validation=True,
    )
    best_validation = float("-inf")
    cumulative_training_steps = 0

    try:
        for iteration in range(args.generations):
            started = time.perf_counter()
            generation = trainer.generation
            agents = trainer.getAgents()
            evaluated: list[tuple[float, Any]] = []
            generation_training_steps = 0

            for agent in agents:
                summary = evaluate_agent(agent, training_seeds, args.max_steps)
                agent.reward(summary.reward, task_name)
                evaluated.append((summary.reward, agent))
                generation_training_steps += sum(
                    episode.steps for episode in summary.episodes
                )
            cumulative_training_steps += generation_training_steps

            evaluated.sort(key=lambda item: item[0], reverse=True)
            current_score, current_agent = evaluated[0]
            rewards = [score for score, _ in evaluated]
            validation = evaluate_agent(
                current_agent,
                validation_seeds,
                args.max_steps,
            )
            if validation.reward > best_validation:
                best_validation = validation.reward
                current_agent.saveToFile(str(checkpoint))

            elapsed = time.perf_counter() - started
            print_progress(
                algorithm="standard_tpg",
                generation=generation + 1,
                total_generations=args.generations,
                rewards=rewards,
                validation=validation,
                best_validation=best_validation,
                population=len(agents),
                seconds=elapsed,
            )
            append_generation_metrics(
                history_path,
                {
                    "algorithm": "standard_tpg",
                    "run_seed": args.run_seed,
                    "generation": generation + 1,
                    "population": len(agents),
                    "train_reward_min": min(rewards),
                    "train_reward_mean": fmean(rewards),
                    "train_reward_max": current_score,
                    "validation_reward": validation.reward,
                    "best_validation_reward": best_validation,
                    "validation_pipes": validation.pipes,
                    "validation_steps": validation.steps,
                    "training_env_steps": generation_training_steps,
                    "cumulative_training_env_steps": cumulative_training_steps,
                    "generation_seconds": elapsed,
                },
            )

            trainer.evolve([task_name])
            trainer.saveToFile(str(run_dir / "latest_trainer.pkl"))
    finally:
        trainer.cleanup()

    agent = Agent.loadAgent(str(checkpoint))
    test_seeds = make_seeds(args.test_seed, args.test_episodes)
    test_summary = evaluate_agent(agent, test_seeds, args.max_steps)
    write_episode_results(
        run_dir / "test_episodes.csv",
        algorithm="standard_tpg",
        training_seed=args.run_seed,
        environment_seeds=test_seeds,
        summary=test_summary,
    )
    print(
        f"test algorithm=standard_tpg seed={args.run_seed} "
        f"reward={test_summary.reward:.3f} pipes={test_summary.pipes:.3f} "
        f"steps={test_summary.steps:.3f}",
        flush=True,
    )


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_is_complete(run_dir: Path, algorithm: str, args: argparse.Namespace) -> bool:
    checkpoint = (
        run_dir / "best.json"
        if algorithm == "weighted_tpg"
        else run_dir / "best_agent.pkl"
    )
    return (
        checkpoint.exists()
        and csv_row_count(run_dir / "history.csv") >= args.generations
        and csv_row_count(run_dir / "test_episodes.csv") >= args.test_episodes
    )


def safely_reset_run(run_dir: Path, output: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_output = output.resolve()
    if resolved_run == resolved_output or resolved_output not in resolved_run.parents:
        raise RuntimeError(f"refusing to remove unsafe run path: {resolved_run}")
    if resolved_run.exists():
        shutil.rmtree(resolved_run)


def stream_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"worker exited with code {return_code}; see {log_path.resolve()}"
        )


def worker_command(
    python: Path,
    algorithm: str,
    seed: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(python.resolve()),
        str(Path(__file__).resolve()),
        "--worker",
        algorithm,
        "--run-seed",
        str(seed),
        "--run-dir",
        str(run_dir.resolve()),
        "--runs",
        "1",
        "--generations",
        str(args.generations),
        "--population",
        str(args.population),
        "--episodes",
        str(args.episodes),
        "--validation-episodes",
        str(args.validation_episodes),
        "--test-episodes",
        str(args.test_episodes),
        "--max-steps",
        str(args.max_steps),
        "--test-seed",
        str(args.test_seed),
    ]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_combined_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_results(
    output: Path,
    seeds: list[int],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    generation_rows: list[dict[str, str]] = []
    episode_rows: list[dict[str, str]] = []
    for algorithm in ALGORITHMS:
        for seed in seeds:
            directory = run_directory(output, algorithm, seed)
            history = directory / "history.csv"
            tests = directory / "test_episodes.csv"
            if history.exists():
                generation_rows.extend(read_csv_rows(history))
            if tests.exists():
                episode_rows.extend(read_csv_rows(tests))
    return generation_rows, episode_rows


def run_level_test_values(
    episode_rows: list[dict[str, str]],
    metric: str,
) -> dict[str, list[float]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in episode_rows:
        key = (row["algorithm"], int(row["training_seed"]))
        grouped.setdefault(key, []).append(float(row[metric]))
    result = {algorithm: [] for algorithm in ALGORITHMS}
    for (algorithm, _seed), values in grouped.items():
        if algorithm in result:
            result[algorithm].append(fmean(values))
    return result


def write_summary(path: Path, episode_rows: list[dict[str, str]]) -> None:
    fields = (
        "algorithm",
        "metric",
        "runs",
        "mean",
        "median",
        "std",
        "ci95_low",
        "ci95_high",
        "minimum",
        "maximum",
    )
    rows: list[dict[str, Any]] = []
    for metric in ("reward", "pipes", "steps"):
        values_by_algorithm = run_level_test_values(episode_rows, metric)
        for algorithm in ALGORITHMS:
            values = values_by_algorithm[algorithm]
            if not values:
                continue
            spread = stdev(values) if len(values) > 1 else 0.0
            margin = 1.96 * spread / math.sqrt(len(values)) if len(values) > 1 else 0.0
            center = fmean(values)
            rows.append(
                {
                    "algorithm": algorithm,
                    "metric": metric,
                    "runs": len(values),
                    "mean": center,
                    "median": median(values),
                    "std": spread,
                    "ci95_low": center - margin,
                    "ci95_high": center + margin,
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def curve_band(
    rows: list[dict[str, str]],
    algorithm: str,
    metric: str,
    x_field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if row["algorithm"] == algorithm:
            grouped.setdefault(int(row["run_seed"]), []).append(row)
    if not grouped:
        return None

    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for seed_rows in grouped.values():
        ordered = sorted(seed_rows, key=lambda row: float(row[x_field]))
        x = np.asarray([float(row[x_field]) for row in ordered], dtype=float)
        y = np.asarray([float(row[metric]) for row in ordered], dtype=float)
        if x.size:
            curves.append((x, y))
    if not curves:
        return None

    if x_field == "generation":
        start = max(curve[0][0] for curve in curves)
        stop = min(curve[0][-1] for curve in curves)
        grid = np.arange(math.ceil(start), math.floor(stop) + 1, dtype=float)
    else:
        start = max(curve[0][0] for curve in curves)
        stop = min(curve[0][-1] for curve in curves)
        grid = np.linspace(start, stop, 200) if stop > start else np.asarray([start])
    if grid.size == 0:
        return None

    samples = np.vstack([np.interp(grid, x, y) for x, y in curves])
    return (
        grid,
        np.median(samples, axis=0),
        np.quantile(samples, 0.25, axis=0),
        np.quantile(samples, 0.75, axis=0),
    )


def plot_comparison(
    path: Path,
    generation_rows: list[dict[str, str]],
    episode_rows: list[dict[str, str]],
    plot_x: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for the comparison figure. Install it with "
            "python -m pip install -r requirements-experiment.txt"
        ) from exc

    colors = {"weighted_tpg": "#0072B2", "standard_tpg": "#D55E00"}
    x_field = (
        "cumulative_training_env_steps"
        if plot_x == "env-steps"
        else "generation"
    )
    x_label = (
        "Cumulative training environment steps"
        if plot_x == "env-steps"
        else "Generation"
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, metric, title, y_label in (
        (axes[0, 0], "validation_reward", "Validation reward", "Mean reward"),
        (axes[0, 1], "validation_pipes", "Validation pipes", "Mean pipes"),
    ):
        for algorithm in ALGORITHMS:
            band = curve_band(generation_rows, algorithm, metric, x_field)
            if band is None:
                continue
            x, center, lower, upper = band
            axis.plot(
                x,
                center,
                color=colors[algorithm],
                label=DISPLAY_NAMES[algorithm],
                linewidth=2,
            )
            axis.fill_between(x, lower, upper, color=colors[algorithm], alpha=0.2)
        axis.set_title(f"{title}: median and IQR across runs")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
        axis.legend()

    for axis, metric, title, y_label in (
        (axes[1, 0], "reward", "Held-out test reward", "Per-run mean reward"),
        (axes[1, 1], "pipes", "Held-out test pipes", "Per-run mean pipes"),
    ):
        values_by_algorithm = run_level_test_values(episode_rows, metric)
        data = [values_by_algorithm[algorithm] for algorithm in ALGORITHMS]
        if all(data):
            boxes = axis.boxplot(
                data,
                positions=[1, 2],
                widths=0.5,
                patch_artist=True,
                labels=[DISPLAY_NAMES[algorithm] for algorithm in ALGORITHMS],
            )
            for box, algorithm in zip(boxes["boxes"], ALGORITHMS):
                box.set_facecolor(colors[algorithm])
                box.set_alpha(0.3)
            for position, algorithm, values in zip([1, 2], ALGORITHMS, data):
                offsets = (
                    np.linspace(-0.08, 0.08, len(values))
                    if len(values) > 1
                    else np.asarray([0.0])
                )
                axis.scatter(
                    position + offsets,
                    values,
                    color=colors[algorithm],
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=3,
                )
        else:
            axis.text(0.5, 0.5, "No complete held-out test data", ha="center", va="center")
            axis.set_xticks([])
        axis.set_title(f"{title}: one point per training seed")
        axis.set_ylabel(y_label)
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Flappy Bird: Standard TPG vs Weighted TPG", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def coordinator(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = [training_seed(args, index) for index in range(args.runs)]

    if not args.plot_only:
        interpreters = {
            "weighted_tpg": args.weighted_python,
            "standard_tpg": args.pytpg_python,
        }
        for algorithm, interpreter in interpreters.items():
            if not interpreter.exists():
                raise FileNotFoundError(
                    f"Python interpreter for {algorithm} not found: {interpreter.resolve()}"
                )

        failures: list[str] = []
        for seed in seeds:
            for algorithm in ALGORITHMS:
                directory = run_directory(output, algorithm, seed)
                if args.overwrite:
                    safely_reset_run(directory, output)
                elif run_is_complete(directory, algorithm, args):
                    print(f"Skipping complete run: {algorithm}, seed={seed}")
                    continue
                elif directory.exists() and any(directory.iterdir()):
                    raise RuntimeError(
                        f"Incomplete run exists at {directory}. Use --overwrite to rerun it."
                    )

                command = worker_command(
                    interpreters[algorithm],
                    algorithm,
                    seed,
                    directory,
                    args,
                )
                print(f"\nStarting {algorithm}, seed={seed}")
                try:
                    stream_command(command, directory / "run.log")
                except Exception as exc:
                    failures.append(f"{algorithm}, seed={seed}: {exc}")
                    if not args.continue_on_error:
                        raise
                    print(f"ERROR: {failures[-1]}", file=sys.stderr)
        if failures:
            (output / "failures.txt").write_text("\n".join(failures), encoding="utf-8")

    generation_rows, episode_rows = collect_results(output, seeds)
    if not generation_rows:
        raise RuntimeError(f"no generation histories found below {output}")
    write_combined_csv(
        output / "all_generations.csv",
        GENERATION_FIELDS,
        generation_rows,
    )
    write_combined_csv(
        output / "all_test_episodes.csv",
        EPISODE_FIELDS,
        episode_rows,
    )
    write_summary(output / "summary.csv", episode_rows)
    plot_comparison(
        output / "comparison.png",
        generation_rows,
        episode_rows,
        args.plot_x,
    )

    configuration = {
        "runs": args.runs,
        "generations": args.generations,
        "population_requested": args.population,
        "episodes_per_agent": args.episodes,
        "validation_episodes": args.validation_episodes,
        "test_episodes": args.test_episodes,
        "max_steps": args.max_steps,
        "training_seeds": seeds,
        "test_seed": args.test_seed,
        "plot_x": args.plot_x,
        "weighted_python": str(args.weighted_python.resolve()),
        "pytpg_python": str(args.pytpg_python.resolve()),
    }
    (output / "config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )
    print(f"\nCombined generation data: {output / 'all_generations.csv'}")
    print(f"Combined held-out data: {output / 'all_test_episodes.csv'}")
    print(f"Statistical summary: {output / 'summary.csv'}")
    print(f"Comparison figure: {output / 'comparison.png'}")


def main() -> None:
    args = parse_args()
    validate_positive(args)
    if args.worker:
        if args.run_seed is None or args.run_dir is None:
            raise ValueError("internal worker mode requires --run-seed and --run-dir")
        if args.worker == "weighted_tpg":
            run_weighted_worker(args)
        else:
            run_standard_worker(args)
        return
    coordinator(args)


if __name__ == "__main__":
    main()
