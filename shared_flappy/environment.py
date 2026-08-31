"""The single Flappy Bird environment used by both TPG implementations."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Optional

import numpy as np


@dataclass
class Pipe:
    x: float
    gap_y: float
    passed: bool = False


class FlappyEnv:
    width = 480
    height = 640
    bird_x = 120.0
    bird_radius = 14.0
    pipe_width = 64.0
    gap_size = 165.0
    pipe_speed = 3.0
    pipe_spacing = 260.0
    gravity = 0.42
    flap_velocity = -7.2

    survival_reward = 0.1
    pipe_reward = 5.0
    collision_penalty = -10.0

    observation_size = 5
    action_count = 2

    def __init__(
        self,
        seed: Optional[int] = None,
        render_mode: str | None = None,
        fps: int = 60,
    ) -> None:
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")

        self.rng = random.Random(seed)
        self.render_mode = render_mode
        self.fps = fps
        self.close_requested = False
        self._pygame: Any = None
        self._screen: Any = None
        self._clock: Any = None
        self._font: Any = None
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        self.bird_y = self.height / 2
        self.velocity = 0.0
        self.score = 0
        self.steps = 0
        self.done = False
        self.close_requested = False
        self.pipes = [Pipe(420.0, self._gap()), Pipe(680.0, self._gap())]
        return self.observation()

    def _gap(self) -> float:
        margin = self.gap_size / 2 + 55
        return self.rng.uniform(margin, self.height - margin)

    def _next_pipe(self) -> Pipe:
        candidates = [
            pipe
            for pipe in self.pipes
            if pipe.x + self.pipe_width >= self.bird_x
        ]
        if candidates:
            return min(candidates, key=lambda pipe: pipe.x)
        return max(self.pipes, key=lambda pipe: pipe.x)

    def observation(self) -> np.ndarray:
        pipe = self._next_pipe()
        return np.array(
            [
                self.bird_y / self.height,
                self.velocity / 10.0,
                (pipe.x - self.bird_x) / self.width,
                (pipe.gap_y - self.gap_size / 2) / self.height,
                (pipe.gap_y + self.gap_size / 2) / self.height,
            ],
            dtype=np.float64,
        )

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done; call reset().")
        if action not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action!r}")

        if action == 1:
            self.velocity = self.flap_velocity
        self.velocity += self.gravity
        self.bird_y += self.velocity
        self.steps += 1
        reward = self.survival_reward

        for pipe in self.pipes:
            pipe.x -= self.pipe_speed
            if not pipe.passed and pipe.x + self.pipe_width < self.bird_x:
                pipe.passed = True
                self.score += 1
                reward += self.pipe_reward

        if self.pipes[0].x + self.pipe_width < 0:
            self.pipes.pop(0)
            self.pipes.append(
                Pipe(self.pipes[-1].x + self.pipe_spacing, self._gap())
            )

        collision = (
            self.bird_y - self.bird_radius <= 0
            or self.bird_y + self.bird_radius >= self.height
        )
        for pipe in self.pipes:
            overlaps = (
                self.bird_x + self.bird_radius >= pipe.x
                and self.bird_x - self.bird_radius <= pipe.x + self.pipe_width
            )
            outside_gap = (
                self.bird_y - self.bird_radius
                < pipe.gap_y - self.gap_size / 2
                or self.bird_y + self.bird_radius
                > pipe.gap_y + self.gap_size / 2
            )
            collision = collision or (overlaps and outside_gap)

        if collision:
            self.done = True
            reward += self.collision_penalty

        return self.observation(), float(reward), self.done, self.info()

    def info(self) -> dict[str, int]:
        return {"score": self.score, "pipes": self.score, "steps": self.steps}

    def render(self) -> bool:
        if self.render_mode != "human":
            return True

        self._ensure_renderer()
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close_requested = True

        screen = self._screen
        screen.fill((118, 202, 224))
        for pipe in self.pipes:
            top = pipe.gap_y - self.gap_size / 2
            bottom = pipe.gap_y + self.gap_size / 2
            pygame.draw.rect(
                screen,
                (55, 180, 75),
                (pipe.x, 0, self.pipe_width, top),
            )
            pygame.draw.rect(
                screen,
                (55, 180, 75),
                (pipe.x, bottom, self.pipe_width, self.height - bottom),
            )
        pygame.draw.circle(
            screen,
            (255, 225, 45),
            (int(self.bird_x), int(self.bird_y)),
            int(self.bird_radius),
        )
        label = self._font.render(f"Pipes: {self.score}", True, (20, 20, 20))
        screen.blit(label, (18, 16))
        pygame.display.flip()
        self._clock.tick(self.fps)
        return not self.close_requested

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None

    def _ensure_renderer(self) -> None:
        if self._screen is not None:
            return
        import pygame

        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird — shared benchmark")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 34)
