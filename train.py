import argparse
from pathlib import Path

from flappy_env import FlappyEnv
from tpg import evolve


def main():
    parser = argparse.ArgumentParser(description="Train a TPG agent on Flappy Bird")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/best.json")
    args = parser.parse_args()

    episode_seeds = [args.seed + i * 997 for i in range(args.episodes)]

    def evaluate(agent):
        total = 0.0
        for seed in episode_seeds:
            env = FlappyEnv(seed)
            state = env.reset(seed)
            for _ in range(args.max_steps):
                state, reward, done, _ = env.step(agent.act(state))
                total += reward
                if done:
                    break
        return total / args.episodes

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def progress(gen, current, overall, best):
        best.save(output)
        print(f"generation={gen + 1:4d} current={current:9.2f} best={overall:9.2f}")

    best, fitness = evolve(evaluate, args.generations, args.population, args.seed, progress)
    best.save(output)
    print(f"saved {output} (fitness={fitness:.2f})")


if __name__ == "__main__":
    main()
