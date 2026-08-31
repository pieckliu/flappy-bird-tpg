"""Import the installed PyTPG package without seeing the parent's tpg.py."""

from __future__ import annotations

import importlib
from pathlib import Path
import os
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
PARENT_REPOSITORY = HERE.parent.resolve()
PARENT_TPG = PARENT_REPOSITORY / "tpg.py"


def _resolved_search_entry(entry: str) -> Path:
    return Path(entry or os.getcwd()).resolve()


def load_pytpg() -> tuple[Any, Any]:
    """Return PyTPG's Trainer and Agent classes.

    The parent repository contains a top-level tpg.py. Removing only that
    repository root from sys.path leaves this standalone directory and normal
    site-packages available while preventing module shadowing.
    """
    existing = sys.modules.get("tpg")
    existing_file = getattr(existing, "__file__", None)
    if existing_file and Path(existing_file).resolve() == PARENT_TPG.resolve():
        del sys.modules["tpg"]

    sys.path[:] = [
        entry
        for entry in sys.path
        if _resolved_search_entry(entry) != PARENT_REPOSITORY
    ]

    try:
        package = importlib.import_module("tpg")
        trainer_module = importlib.import_module("tpg.trainer")
        agent_module = importlib.import_module("tpg.agent")
    except (ImportError, ModuleNotFoundError) as exc:
        detail = str(exc)
        if "NINF" in detail and "numpy" in detail:
            raise RuntimeError(
                "This PyTPG revision requires NumPy 1.x. Reinstall the isolated "
                "dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        raise RuntimeError(
            "PyTPG is not installed in this interpreter. Create the dedicated "
            "virtual environment and run: python -m pip install -r requirements.txt"
        ) from exc

    origin = Path(getattr(package, "__file__", "")).resolve()
    if origin == PARENT_TPG.resolve():
        raise RuntimeError(
            f"Imported the parent project's {PARENT_TPG} instead of PyTPG. "
            "Run the script with the dedicated pytpg_flappy virtual environment."
        )

    # PyTPG revisions have exposed these loaders both as class attributes and
    # as module-level functions. Normalize the API for the local entry points.
    if not hasattr(trainer_module.Trainer, "loadTrainer"):
        trainer_module.Trainer.loadTrainer = staticmethod(trainer_module.loadTrainer)
    if not hasattr(agent_module.Agent, "loadAgent"):
        agent_module.Agent.loadAgent = staticmethod(agent_module.loadAgent)

    return trainer_module.Trainer, agent_module.Agent
