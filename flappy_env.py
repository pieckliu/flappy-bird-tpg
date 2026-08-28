from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

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
    gravity = 0.42
    flap_velocity = -7.2

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        self.bird_y = self.height / 2
        self.velocity = 0.0
        self.score = 0
        self.steps = 0
        self.done = False
        self.pipes = [Pipe(420.0, self._gap()), Pipe(680.0, self._gap())]
        return self.observation()

    def _gap(self) -> float:
        margin = self.gap_size / 2 + 55
        return self.rng.uniform(margin, self.height - margin)

    def _next_pipe(self) -> Pipe:
        return min((p for p in self.pipes if p.x + self.pipe_width >= self.bird_x), key=lambda p: p.x)

    def observation(self) -> np.ndarray:
        pipe = self._next_pipe()
        return np.array([
            self.bird_y / self.height,
            self.velocity / 10.0,
            (pipe.x - self.bird_x) / self.width,
            (pipe.gap_y - self.gap_size / 2) / self.height,
            (pipe.gap_y + self.gap_size / 2) / self.height,
        ], dtype=np.float64)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done; call reset().")
        if action == 1:
            self.velocity = self.flap_velocity
        self.velocity += self.gravity
        self.bird_y += self.velocity
        self.steps += 1
        reward = 0.1

        for pipe in self.pipes:
            pipe.x -= self.pipe_speed
            if not pipe.passed and pipe.x + self.pipe_width < self.bird_x:
                pipe.passed = True
                self.score += 1
                reward += 5.0
        if self.pipes[0].x + self.pipe_width < 0:
            self.pipes.pop(0)
            self.pipes.append(Pipe(self.pipes[-1].x + 260.0, self._gap()))

        collision = self.bird_y - self.bird_radius <= 0 or self.bird_y + self.bird_radius >= self.height
        for pipe in self.pipes:
            overlaps = self.bird_x + self.bird_radius >= pipe.x and self.bird_x - self.bird_radius <= pipe.x + self.pipe_width
            outside_gap = self.bird_y - self.bird_radius < pipe.gap_y - self.gap_size / 2 or self.bird_y + self.bird_radius > pipe.gap_y + self.gap_size / 2
            collision = collision or (overlaps and outside_gap)
        if collision:
            self.done = True
            reward -= 10.0
        return self.observation(), reward, self.done, {"score": self.score, "steps": self.steps}
