"""Train PyTPG on the same environment and evaluator as the compact TPG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import fmean
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared_flappy import FlappyEnv, evaluate_agent, make_seeds
from pytpg_loader import load_pytpg


TASK_NAME = "flappy-bird-shared"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "shared_env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "generations",
        "population",
        "episodes",
        "validation_episodes",
        "max_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")


def create_trainer(Trainer: Any, population: int) -> Any:
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


def write_metadata(
    output: Path,
    args: argparse.Namespace,
    trainer: Any,
    best_score: float,
    minimum: float,
    average: float,
    maximum: float,
) -> None:
    metadata = {
        "algorithm": "pytpg",
        "environment": "shared_flappy.FlappyEnv",
        "task": TASK_NAME,
        "generation": trainer.generation,
        "best_score": best_score,
        "last_generation": {
            "minimum": minimum,
            "average": average,
            "maximum": maximum,
        },
        "configuration": {
            "training_seed": args.seed,
            "episodes_per_agent": args.episodes,
            "validation_episodes": args.validation_episodes,
            "max_steps": args.max_steps,
            "population_requested": args.population,
            "observation_size": FlappyEnv.observation_size,
            "actions": FlappyEnv.action_count,
        },
    }
    (output / "training.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    Trainer, _ = load_pytpg()

    if args.resume:
        trainer = Trainer.loadTrainer(str(args.resume.resolve()))
        if trainer.inputSize != FlappyEnv.observation_size:
            raise ValueError(
                "The resumed trainer was created for a different observation "
                "space. Start a new shared-environment run instead."
            )
        print(
            f"Resumed generation {trainer.generation} from "
            f"{args.resume.resolve()}"
        )
    else:
        trainer = create_trainer(Trainer, args.population)

    best_validation = float("-inf")
    metadata_path = args.output / "training.json"
    if args.resume and metadata_path.exists():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            best_validation = float(previous.get("best_score", best_validation))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            print("Could not read the previous best score; starting a new record.")

    training_seeds = make_seeds(args.seed, args.episodes)
    validation_seeds = make_seeds(
        args.seed,
        args.validation_episodes,
        validation=True,
    )

    try:
        for iteration in range(args.generations):
            started = time.perf_counter()
            generation = trainer.generation
            agents = trainer.getAgents()
            evaluated = []

            for agent in agents:
                summary = evaluate_agent(agent, training_seeds, args.max_steps)
                agent.reward(summary.reward, TASK_NAME)
                evaluated.append((summary.reward, agent))

            evaluated.sort(key=lambda item: item[0], reverse=True)
            current_score, current_agent = evaluated[0]
            rewards = [score for score, _ in evaluated]
            minimum = min(rewards)
            average = fmean(rewards)
            maximum = max(rewards)

            validation = evaluate_agent(
                current_agent,
                validation_seeds,
                args.max_steps,
            )
            if validation.reward > best_validation:
                best_validation = validation.reward
                current_agent.saveToFile(str(args.output / "best_agent.pkl"))

            elapsed = time.perf_counter() - started
            print(
                f"algorithm=pytpg "
                f"generation={generation + 1:04d} "
                f"iteration={iteration + 1:04d}/{args.generations:04d} "
                f"agents={len(agents):04d} "
                f"min={minimum:8.2f} "
                f"avg={average:8.2f} "
                f"max={current_score:8.2f} "
                f"val={validation.reward:8.2f} "
                f"best={best_validation:8.2f} "
                f"val_pipes={validation.pipes:7.2f} "
                f"val_steps={validation.steps:8.2f} "
                f"seconds={elapsed:7.2f}"
            )

            trainer.evolve([TASK_NAME])
            trainer.saveToFile(str(args.output / "latest_trainer.pkl"))
            write_metadata(
                args.output,
                args,
                trainer,
                best_validation,
                minimum,
                average,
                maximum,
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted; saving the current trainer.")
        trainer.saveToFile(str(args.output / "latest_trainer.pkl"))
    finally:
        trainer.cleanup()

    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()