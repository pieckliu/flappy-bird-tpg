import argparse
import sys

import pygame

from flappy_env import FlappyEnv
from tpg import TPGAgent


def main():
    parser = argparse.ArgumentParser(description="Watch a trained TPG agent")
    parser.add_argument("--model", default="checkpoints/best.json")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    agent, env = TPGAgent.load(args.model), FlappyEnv(args.seed)
    state = env.reset(args.seed)
    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Flappy Bird — TPG")
    clock, font = pygame.time.Clock(), pygame.font.Font(None, 34)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
        state, _, done, info = env.step(agent.act(state))
        screen.fill((118, 202, 224))
        for pipe in env.pipes:
            top = pipe.gap_y - env.gap_size / 2
            bottom = pipe.gap_y + env.gap_size / 2
            pygame.draw.rect(screen, (55, 180, 75), (pipe.x, 0, env.pipe_width, top))
            pygame.draw.rect(screen, (55, 180, 75), (pipe.x, bottom, env.pipe_width, env.height - bottom))
        pygame.draw.circle(screen, (255, 225, 45), (int(env.bird_x), int(env.bird_y)), int(env.bird_radius))
        screen.blit(font.render(f"Score: {info['score']}", True, (20, 20, 20)), (18, 16))
        pygame.display.flip()
        clock.tick(args.fps)
        if done:
            state = env.reset(args.seed)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        sys.exit("模型不存在，请先运行 train.py。")
