import argparse
from pathlib import Path
from statistics import fmean
import time

from shared_flappy import evaluate_agent, make_seeds
from tpg import evolve


def main():
    parser = argparse.ArgumentParser(
        description="Train the compact TPG on the shared Flappy Bird benchmark"
    )
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/shared_env/best.json")
    args = parser.parse_args()

    for name in (
        "generations",
        "population",
        "episodes",
        "validation_episodes",
        "max_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")

    training_seeds = make_seeds(args.seed, args.episodes)
    validation_seeds = make_seeds(
        args.seed,
        args.validation_episodes,
        validation=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_validation = float("-inf")
    generation_started = time.perf_counter()

    def evaluate(agent):
        return evaluate_agent(agent, training_seeds, args.max_steps).reward

    def progress(generation, scored, _training_best, _best_agent):
        nonlocal best_validation, generation_started
        current_score, current_agent = scored[0]
        validation = evaluate_agent(
            current_agent,
            validation_seeds,
            args.max_steps,
        )
        if validation.reward > best_validation:
            best_validation = validation.reward
            current_agent.save(output)

        rewards = [score for score, _ in scored]
        elapsed = time.perf_counter() - generation_started
        print(
            f"algorithm=compact "
            f"generation={generation + 1:04d} "
            f"iteration={generation + 1:04d}/{args.generations:04d} "
            f"agents={len(scored):04d} "
            f"min={min(rewards):8.2f} "
            f"avg={fmean(rewards):8.2f} "
            f"max={current_score:8.2f} "
            f"val={validation.reward:8.2f} "
            f"best={best_validation:8.2f} "
            f"val_pipes={validation.pipes:7.2f} "
            f"val_steps={validation.steps:8.2f} "
            f"seconds={elapsed:7.2f}"
        )
        generation_started = time.perf_counter()

    evolve(
        evaluate,
        generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        on_scored=progress,
    )
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()