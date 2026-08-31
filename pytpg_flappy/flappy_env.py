"""Compatibility aliases for the repository-wide shared Flappy environment."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared_flappy import FlappyEnv, Pipe

FlappyBirdEnv = FlappyEnv

__all__ = ["FlappyBirdEnv", "FlappyEnv", "Pipe"]