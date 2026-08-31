"""One evaluation implementation shared by the compact TPG and PyTPG."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable

import numpy as np

from .environment import FlappyEnv


SEED_STRIDE = 997
VALIDATION_OFFSET = 50_000_000


@dataclass(frozen=True)
class EpisodeResult:
    reward: float
    pipes: int
    steps: int
    terminated: bool
    cancelled: bool = False


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: tuple[EpisodeResult, ...]

    @property
    def reward(self) -> float:
        return fmean(episode.reward for episode in self.episodes)

    @property
    def pipes(self) -> float:
        return fmean(episode.pipes for episode in self.episodes)

    @property
    def steps(self) -> float:
        return fmean(episode.steps for episode in self.episodes)


def make_seeds(base_seed: int, count: int, validation: bool = False) -> list[int]:
    if count <= 0:
        raise ValueError("seed count must be greater than zero")
    offset = VALIDATION_OFFSET if validation else 0
    return [base_seed + offset + index * SEED_STRIDE for index in range(count)]


def run_episode(
    agent: Any,
    seed: int,
    max_steps: int,
    render: bool = False,
    fps: int = 60,
) -> EpisodeResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero")

    env = FlappyEnv(seed=seed, render_mode="human" if render else None, fps=fps)
    state = env.reset(seed)
    reset_registers = getattr(agent, "zeroRegisters", None)
    if callable(reset_registers):
        reset_registers()

    total_reward = 0.0
    cancelled = False
    done = False
    info = env.info()

    try:
        if render and not env.render():
            cancelled = True

        while not cancelled and not done and info["steps"] < max_steps:
            observation = np.ascontiguousarray(state, dtype=np.float64)
            action = int(agent.act(observation))
            state, reward, done, info = env.step(action)
            total_reward += reward
            if render and not env.render():
                cancelled = True
    finally:
        env.close()

    return EpisodeResult(
        reward=float(total_reward),
        pipes=int(info["pipes"]),
        steps=int(info["steps"]),
        terminated=done,
        cancelled=cancelled,
    )


def evaluate_agent(
    agent: Any,
    seeds: Iterable[int],
    max_steps: int,
) -> EvaluationSummary:
    seed_values = list(seeds)
    if not seed_values:
        raise ValueError("at least one evaluation seed is required")
    return EvaluationSummary(
        tuple(run_episode(agent, seed, max_steps) for seed in seed_values)
    )
