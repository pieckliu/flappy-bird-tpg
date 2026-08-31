"""Compatibility wrapper around the repository-wide shared episode runner."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared_flappy import run_episode as run_shared_episode


def run_episode(
    agent: Any,
    env: Any,
    seed: int,
    render: bool = False,
) -> dict[str, float | int | bool]:
    max_steps = int(getattr(env, "max_steps", 6000))
    fps = int(getattr(env, "fps", 60))
    result = run_shared_episode(
        agent,
        seed=seed,
        max_steps=max_steps,
        render=render,
        fps=fps,
    )
    return {
        "reward": result.reward,
        "pipes": result.pipes,
        "steps": result.steps,
        "cancelled": result.cancelled,
    }