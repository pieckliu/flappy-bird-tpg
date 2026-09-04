"""Plot mean population fitness by generation for the TPG comparison."""

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


ALGORITHM_LABELS = {
    "weighted_tpg": "Weighted TPG",
    "standard_tpg": "Standard TPG (PyTPG)",
}

ALGORITHM_COLORS = {
    "weighted_tpg": "#0072B2",
    "standard_tpg": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot generation against mean fitness for both TPG algorithms."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/tpg_comparison_p10_g50/all_generations.csv"),
        help="Path to all_generations.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults beside the input CSV.",
    )
    return parser.parse_args()


def load_fitness(path: Path) -> dict[str, dict[int, list[float]]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"algorithm", "generation", "train_reward_mean"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

        for row in reader:
            grouped[row["algorithm"]][int(row["generation"])].append(
                float(row["train_reward_mean"])
            )

    if not grouped:
        raise ValueError(f"No training data found in {path}")
    return grouped


def summarize(values: list[float]) -> tuple[float, float]:
    """Return the mean and normal-approximation 95% confidence interval half-width."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return mean, 1.96 * standard_error


def write_summary(
    path: Path, grouped: dict[str, dict[int, list[float]]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["algorithm", "generation", "runs", "mean_fitness", "ci95_lower", "ci95_upper"]
        )
        for algorithm in sorted(grouped):
            for generation in sorted(grouped[algorithm]):
                values = grouped[algorithm][generation]
                mean, half_width = summarize(values)
                writer.writerow(
                    [
                        algorithm,
                        generation,
                        len(values),
                        mean,
                        mean - half_width,
                        mean + half_width,
                    ]
                )


def plot(
    output: Path, grouped: dict[str, dict[int, list[float]]]
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), dpi=180)

    preferred_order = ["weighted_tpg", "standard_tpg"]
    algorithms = [name for name in preferred_order if name in grouped]
    algorithms.extend(name for name in grouped if name not in algorithms)

    for algorithm in algorithms:
        generations = sorted(grouped[algorithm])
        summaries = [summarize(grouped[algorithm][g]) for g in generations]
        means = [item[0] for item in summaries]
        half_widths = [item[1] for item in summaries]
        lower = [mean - width for mean, width in summaries]
        upper = [mean + width for mean, width in summaries]
        color = ALGORITHM_COLORS.get(algorithm)
        label = ALGORITHM_LABELS.get(algorithm, algorithm)

        axis.plot(
            generations,
            means,
            color=color,
            linewidth=2.4,
            label=label,
        )
        axis.fill_between(
            generations,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    all_generations = [g for values in grouped.values() for g in values]
    axis.set_xlim(min(all_generations), max(all_generations))
    axis.axhline(0, color="#555555", linewidth=0.8, alpha=0.65)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Mean fitness (episode reward)")
    axis.set_title("Flappy Bird: Mean Fitness by Generation")
    axis.grid(True, color="#B0B0B0", alpha=0.28)
    axis.legend(title="Mean across training seeds", frameon=True)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name("mean_fitness_by_generation.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = load_fitness(input_path)
    plot(output_path, grouped)
    summary_path = output_path.with_suffix(".csv")
    write_summary(summary_path, grouped)

    print(f"Chart: {output_path}")
    print(f"Summary: {summary_path}")
    for algorithm in ("weighted_tpg", "standard_tpg"):
        if algorithm not in grouped:
            continue
        last_generation = max(grouped[algorithm])
        final_mean, final_ci = summarize(grouped[algorithm][last_generation])
        print(
            f"{ALGORITHM_LABELS[algorithm]} generation {last_generation}: "
            f"mean={final_mean:.3f}, 95% CI=[{final_mean-final_ci:.3f}, "
            f"{final_mean+final_ci:.3f}]"
        )


if __name__ == "__main__":
    main()
