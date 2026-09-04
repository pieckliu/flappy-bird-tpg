import csv
import tempfile
import unittest
from pathlib import Path

from shared_flappy.evaluation import EpisodeResult, EvaluationSummary
from shared_flappy.metrics import (
    GENERATION_FIELDS,
    append_generation_metrics,
    prepare_generation_history,
    write_episode_results,
)


class MetricsTests(unittest.TestCase):
    def test_generation_and_episode_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = prepare_generation_history(root / "history.csv")
            values = {field: index for index, field in enumerate(GENERATION_FIELDS)}
            values["algorithm"] = "weighted_tpg"
            append_generation_metrics(history, values)

            with history.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["algorithm"], "weighted_tpg")

            summary = EvaluationSummary(
                (
                    EpisodeResult(1.5, 2, 30, True),
                    EpisodeResult(2.5, 3, 40, True),
                )
            )
            episodes = write_episode_results(
                root / "episodes.csv",
                algorithm="weighted_tpg",
                training_seed=42,
                environment_seeds=[100, 101],
                summary=summary,
            )
            with episodes.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["environment_seed"] for row in rows], ["100", "101"])
            self.assertEqual([row["pipes"] for row in rows], ["2", "3"])


if __name__ == "__main__":
    unittest.main()
