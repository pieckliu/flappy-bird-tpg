import tempfile
import unittest
from pathlib import Path

import numpy as np

from flappy_env import FlappyEnv
from tpg import TPGAgent


class ProjectTests(unittest.TestCase):
    def test_environment_is_deterministic(self):
        a, b = FlappyEnv(7), FlappyEnv(7)
        for action in [0, 1, 0, 0, 1]:
            oa, ra, da, ia = a.step(action)
            ob, rb, db, ib = b.step(action)
            np.testing.assert_allclose(oa, ob)
            self.assertEqual((ra, da, ia), (rb, db, ib))

    def test_agent_returns_valid_action_and_roundtrips(self):
        agent = TPGAgent.random_agent(np.random.default_rng(1))
        state = FlappyEnv(1).reset()
        self.assertIn(agent.act(state), (0, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            agent.save(path)
            self.assertEqual(TPGAgent.load(path).act(state), agent.act(state))


if __name__ == "__main__":
    unittest.main()
