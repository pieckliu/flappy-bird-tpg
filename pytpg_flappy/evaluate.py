"""Evaluate a PyTPG agent with the repository-wide shared evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean, median, pstdev
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared_flappy import evaluate_agent, make_seeds
from pytpg_loader import load_pytpg


DEFAULT_AGENT = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "shared_env"
    / "best_agent.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=6000)
    return parser.parse_args()


def describe(name: str, values: list[float | int]) -> None:
    numeric = [float(value) for value in values]
    print(
        f"{name}: mean={fmean(numeric):.3f} "
        f"median={median(numeric):.3f} "
        f"std={pstdev(numeric):.3f} "
        f"min={min(numeric):.3f} max={max(numeric):.3f}"
    )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero")

    agent_path = args.agent.resolve()
    if not agent_path.exists():
        raise FileNotFoundError(
            f"Agent checkpoint not found: {agent_path}. Run train.py first."
        )

    _, Agent = load_pytpg()
    agent = Agent.loadAgent(str(agent_path))
    seeds = make_seeds(args.seed, args.episodes)
    summary = evaluate_agent(agent, seeds, args.max_steps)

    print(f"agent: {agent_path}")
    print(f"episodes: {args.episodes}, first_seed: {seeds[0]}, last_seed: {seeds[-1]}")
    describe("reward", [episode.reward for episode in summary.episodes])
    describe("pipes", [episode.pipes for episode in summary.episodes])
    describe("steps", [episode.steps for episode in summary.episodes])


if __name__ == "__main__":
    main()