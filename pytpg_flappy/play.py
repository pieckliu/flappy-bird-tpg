"""Render a PyTPG agent on the shared Flappy Bird benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared_flappy import run_episode
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
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=6000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_path = args.agent.resolve()
    if not agent_path.exists():
        raise FileNotFoundError(
            f"Agent checkpoint not found: {agent_path}. Run train.py first."
        )

    _, Agent = load_pytpg()
    agent = Agent.loadAgent(str(agent_path))

    for episode in range(args.episodes):
        result = run_episode(
            agent,
            seed=args.seed + episode * 997,
            max_steps=args.max_steps,
            render=True,
            fps=args.fps,
        )
        print(
            f"episode={episode + 1} seed={args.seed + episode * 997} "
            f"pipes={result.pipes} reward={result.reward:.2f} "
            f"steps={result.steps}"
        )
        if result.cancelled:
            break


if __name__ == "__main__":
    main()