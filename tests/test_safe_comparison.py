import unittest

import run_comparison_safe
from shared_flappy import evaluate_agent


class BrokenGraphAgent:
    def zeroRegisters(self):
        pass

    def act(self, _state):
        raise ValueError("max() arg is an empty sequence")


class SafeComparisonTests(unittest.TestCase):
    def test_empty_team_uses_neutral_action(self):
        before = run_comparison_safe._dead_end_fallbacks
        summary = evaluate_agent(BrokenGraphAgent(), [7], max_steps=100)
        self.assertEqual(len(summary.episodes), 1)
        self.assertGreater(run_comparison_safe._dead_end_fallbacks, before)


if __name__ == "__main__":
    unittest.main()
