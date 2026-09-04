"""Narrow compatibility hook for the comparison script's Matplotlib API."""

from __future__ import annotations

from functools import wraps
import inspect
from pathlib import Path
import sys


if Path(sys.argv[0]).name == "compare_experiment.py":
    try:
        from matplotlib.axes import Axes

        signature = inspect.signature(Axes.boxplot)
        if "labels" not in signature.parameters and "tick_labels" in signature.parameters:
            original_boxplot = Axes.boxplot

            @wraps(original_boxplot)
            def compatible_boxplot(self, *args, labels=None, **kwargs):
                if labels is not None and "tick_labels" not in kwargs:
                    kwargs["tick_labels"] = labels
                return original_boxplot(self, *args, **kwargs)

            Axes.boxplot = compatible_boxplot
    except ImportError:
        # The isolated PyTPG interpreter does not need Matplotlib because only
        # the coordinator process builds the final figure.
        pass
