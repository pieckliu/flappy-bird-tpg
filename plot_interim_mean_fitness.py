"""Plot a balanced interim mean-fitness comparison from completed paired seeds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALGORITHMS = ("weighted_tpg", "standard_tpg")
LABELS = {
    "weighted_tpg": "Weighted TPG",
    "standard_tpg": "Standard TPG (PyTPG)",
}
COLORS = {
    "weighted_tpg": "#0072B2",
    "standard_tpg": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("experiments/tpg_comparison"),
    )
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def completed_seed_histories(
    experiment: Path,
    generations: int,
) -> tuple[list[str], dict[str, dict[str, list[dict[str, str]]]]]:
    histories: dict[str, dict[str, list[dict[str, str]]]] = {
        algorithm: {} for algorithm in ALGORITHMS
    }
    for algorithm in ALGORITHMS:
        algorithm_dir = experiment / algorithm
        for seed_dir in sorted(algorithm_dir.glob("seed_[0-9]*")):
            history_path = seed_dir / "history.csv"
            test_path = seed_dir / "test_episodes.csv"
            if not history_path.is_file() or not test_path.is_file():
                continue
            rows = read_history(history_path)
            test_rows = read_history(test_path)
            if len(rows) >= generations and len(test_rows) >= 100:
                histories[algorithm][seed_dir.name] = rows[:generations]

    paired_seeds = sorted(
        set(histories["weighted_tpg"]).intersection(histories["standard_tpg"])
    )
    if not paired_seeds:
        raise RuntimeError("No completed paired seeds are available yet")
    return paired_seeds, histories


def aggregate(
    paired_seeds: list[str],
    histories: dict[str, dict[str, list[dict[str, str]]]],
) -> dict[str, dict[int, list[float]]]:
    grouped: dict[str, dict[int, list[float]]] = {
        algorithm: defaultdict(list) for algorithm in ALGORITHMS
    }
    for algorithm in ALGORITHMS:
        for seed in paired_seeds:
            for row in histories[algorithm][seed]:
                grouped[algorithm][int(row["generation"])].append(
                    float(row["train_reward_mean"])
                )
    return grouped


def summarize(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def write_summary(
    path: Path,
    grouped: dict[str, dict[int, list[float]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "algorithm",
                "generation",
                "paired_seeds",
                "mean_fitness",
                "ci95_lower",
                "ci95_upper",
            ]
        )
        for algorithm in ALGORITHMS:
            for generation in sorted(grouped[algorithm]):
                values = grouped[algorithm][generation]
                mean, interval = summarize(values)
                writer.writerow(
                    [
                        algorithm,
                        generation,
                        len(values),
                        mean,
                        mean - interval,
                        mean + interval,
                    ]
                )


def make_plot(
    path: Path,
    grouped: dict[str, dict[int, list[float]]],
    paired_count: int,
    expected_count: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), dpi=180)
    for algorithm in ALGORITHMS:
        generations = sorted(grouped[algorithm])
        summaries = [summarize(grouped[algorithm][g]) for g in generations]
        means = [value[0] for value in summaries]
        lower = [mean - interval for mean, interval in summaries]
        upper = [mean + interval for mean, interval in summaries]
        axis.plot(
            generations,
            means,
            color=COLORS[algorithm],
            linewidth=2.3,
            label=LABELS[algorithm],
        )
        axis.fill_between(
            generations,
            lower,
            upper,
            color=COLORS[algorithm],
            alpha=0.16,
            linewidth=0,
        )

    axis.axhline(0, color="#555555", linewidth=0.8, alpha=0.65)
    axis.set_xlim(1, max(grouped["weighted_tpg"]))
    axis.set_xlabel("Generation")
    axis.set_ylabel("Mean fitness (episode reward)")
    axis.set_title(
        "Flappy Bird: Interim Mean Fitness by Generation\n"
        f"Balanced snapshot — {paired_count}/{expected_count} paired seeds complete"
    )
    axis.grid(True, color="#B0B0B0", alpha=0.28)
    axis.legend(title="Mean across completed paired seeds", frameon=True)
    figure.text(
        0.5,
        0.01,
        "Lines show the cross-seed mean; shaded bands show approximate 95% confidence intervals. "
        "Incomplete and unpaired runs are excluded.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    experiment = args.experiment.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else experiment / "mean_fitness_by_generation_interim.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    paired_seeds, histories = completed_seed_histories(
        experiment,
        args.generations,
    )
    grouped = aggregate(paired_seeds, histories)
    make_plot(output, grouped, len(paired_seeds), args.expected_seeds)
    summary_path = output.with_suffix(".csv")
    write_summary(summary_path, grouped)

    print(f"Completed paired seeds: {len(paired_seeds)}/{args.expected_seeds}")
    print("Seeds: " + ", ".join(seed.removeprefix("seed_") for seed in paired_seeds))
    print(f"Chart: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
