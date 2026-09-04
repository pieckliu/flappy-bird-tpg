"""Record the trained weighted TPG agent playing Flappy Bird to MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from flappy_env import FlappyEnv
from tpg import TPGAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/shared_env/best.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/flappy_bird_play.mp4"),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a real-time Pygame window while recording",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must use the .mp4 extension")
    return args


def draw_frame(surface: pygame.Surface, env: FlappyEnv, font, info) -> None:
    surface.fill((118, 202, 224))
    for pipe in env.pipes:
        top = pipe.gap_y - env.gap_size / 2
        bottom = pipe.gap_y + env.gap_size / 2
        pygame.draw.rect(
            surface,
            (55, 180, 75),
            (pipe.x, 0, env.pipe_width, top),
        )
        pygame.draw.rect(
            surface,
            (55, 180, 75),
            (pipe.x, bottom, env.pipe_width, env.height - bottom),
        )
    pygame.draw.circle(
        surface,
        (255, 225, 45),
        (int(env.bird_x), int(env.bird_y)),
        int(env.bird_radius),
    )
    label = font.render(f"Score: {info['score']}", True, (20, 20, 20))
    surface.blit(label, (18, 16))


def append_frame(writer, surface: pygame.Surface) -> None:
    # pygame.surfarray uses (width, height, RGB), while video uses
    # (height, width, RGB).
    writer.append_data(pygame.surfarray.array3d(surface).swapaxes(0, 1))


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run train.py first."
        )

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Recording requires imageio and imageio-ffmpeg. Run: "
            "python -m pip install imageio imageio-ffmpeg"
        ) from exc

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    agent = TPGAgent.load(args.model)
    env = FlappyEnv(args.seed)

    pygame.init()
    if args.show:
        surface = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Flappy Bird — TPG recording")
    else:
        surface = pygame.Surface((env.width, env.height))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 34)
    writer = imageio.get_writer(
        str(output),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    cancelled = False
    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index * 997
            state = env.reset(episode_seed)
            info = env.info()
            draw_frame(surface, env, font, info)
            append_frame(writer, surface)

            total_reward = 0.0
            for _ in range(args.max_steps):
                if args.show:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            cancelled = True
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            cancelled = True
                    if cancelled:
                        break

                state, reward, done, info = env.step(agent.act(state))
                total_reward += reward
                draw_frame(surface, env, font, info)
                append_frame(writer, surface)

                if args.show:
                    pygame.display.flip()
                    clock.tick(args.fps)
                if done:
                    break

            print(
                f"episode={episode_index + 1} seed={episode_seed} "
                f"pipes={info['pipes']} reward={total_reward:.2f} "
                f"steps={info['steps']}"
            )
            if cancelled:
                break
    finally:
        writer.close()
        pygame.quit()

    print(f"Video saved to: {output}")


if __name__ == "__main__":
    main()
