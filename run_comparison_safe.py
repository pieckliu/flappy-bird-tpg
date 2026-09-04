"""Robust entry point for the TPG comparison experiment.

In addition to the Matplotlib compatibility handled by ``run_comparison``, this
entry point resolves one upstream PyTPG graph dead end. PyTPG can occasionally
reach a team whose non-atomic learners all point to already visited teams and
whose atomic learner set is empty. Upstream then calls ``max([])``. The compact
TPG already resolves cyclic/depth-invalid paths as action 0, so this adapter
uses the same action for this exact PyTPG exception and re-raises every other
error.
"""

from __future__ import annotations

import atexit
from functools import wraps
import inspect
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
_dead_end_fallbacks = 0


def patch_matplotlib_boxplot_api() -> None:
    try:
        from matplotlib.axes import Axes
    except ImportError:
        return

    signature = inspect.signature(Axes.boxplot)
    if "labels" in signature.parameters or "tick_labels" not in signature.parameters:
        return
    original_boxplot = Axes.boxplot

    @wraps(original_boxplot)
    def compatible_boxplot(self, *args, labels=None, **kwargs):
        if labels is not None and "tick_labels" not in kwargs:
            kwargs["tick_labels"] = labels
        return original_boxplot(self, *args, **kwargs)

    Axes.boxplot = compatible_boxplot


def patch_shared_episode_runner() -> None:
    import shared_flappy.evaluation as evaluation

    original_run_episode = evaluation.run_episode

    class SafeGraphAgent:
        def __init__(self, agent: Any):
            self.agent = agent

        def zeroRegisters(self) -> None:
            reset = getattr(self.agent, "zeroRegisters", None)
            if callable(reset):
                reset()

        def act(self, state: Any) -> int:
            global _dead_end_fallbacks
            try:
                return int(self.agent.act(state))
            except ValueError as exc:
                if str(exc) != "max() arg is an empty sequence":
                    raise
                _dead_end_fallbacks += 1
                return 0

    @wraps(original_run_episode)
    def safe_run_episode(agent: Any, *args, **kwargs):
        return original_run_episode(SafeGraphAgent(agent), *args, **kwargs)

    evaluation.run_episode = safe_run_episode


def report_dead_end_fallbacks() -> None:
    if _dead_end_fallbacks:
        print(
            "warning=standard_tpg_empty_team_fallback "
            f"count={_dead_end_fallbacks}",
            flush=True,
        )


# Child interpreters launched by the coordinator must execute this same entry
# point, not the older compatibility-only wrapper.
os.environ["TPG_COMPARISON_SAFE_ENTRY"] = str(Path(__file__).resolve())
patch_matplotlib_boxplot_api()
patch_shared_episode_runner()
atexit.register(report_dead_end_fallbacks)

import compare_experiment


_original_worker_command = compare_experiment.worker_command


def safe_worker_command(*args, **kwargs):
    command = _original_worker_command(*args, **kwargs)
    command[1] = os.environ["TPG_COMPARISON_SAFE_ENTRY"]
    return command


compare_experiment.worker_command = safe_worker_command


if __name__ == "__main__":
    compare_experiment.main()
