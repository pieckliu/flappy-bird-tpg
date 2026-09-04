"""Stable public entry point for the TPG comparison experiment."""

from __future__ import annotations

from functools import wraps
import inspect


def patch_matplotlib_boxplot_api() -> None:
    """Bridge Matplotlib's labels -> tick_labels rename across 3.x releases."""
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


patch_matplotlib_boxplot_api()

from compare_experiment import main


if __name__ == "__main__":
    main()
