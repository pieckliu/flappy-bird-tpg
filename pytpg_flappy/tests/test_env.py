import unittest

import numpy as np

from flappy_env import FlappyBirdEnv
from shared_flappy import FlappyEnv, evaluate_agent, make_seeds


class ConstantAgent:
    def __init__(self, action=0):
        self.action = action
        self.reset_count = 0

    def act(self, _state):
        return self.action

    def zeroRegisters(self):
        self.reset_count += 1


class SharedFlappyEnvTests(unittest.TestCase):
    def test_subproject_alias_is_the_shared_environment(self):
        self.assertIs(FlappyBirdEnv, FlappyEnv)

    def test_reset_is_deterministic_for_a_seed(self):
        first = FlappyBirdEnv(123)
        second = FlappyBirdEnv(123)

        np.testing.assert_array_equal(first.reset(123), second.reset(123))
        self.assertEqual(
            [pipe.gap_y for pipe in first.pipes],
            [pipe.gap_y for pipe in second.pipes],
        )

    def test_observation_has_shared_shape_and_type(self):
        env = FlappyBirdEnv(1)
        observation = env.reset(1)

        self.assertEqual(observation.shape, (FlappyBirdEnv.observation_size,))
        self.assertEqual(observation.dtype, np.float64)
        self.assertEqual(FlappyBirdEnv.observation_size, 5)

    def test_flap_moves_velocity_upward(self):
        env = FlappyBirdEnv(1)
        env.reset(1)

        _, _, done, _ = env.step(1)

        self.assertLess(env.velocity, 0)
        self.assertFalse(done)

    def test_invalid_action_is_rejected(self):
        env = FlappyBirdEnv(1)
        with self.assertRaises(ValueError):
            env.step(2)

    def test_shared_evaluator_resets_agent_each_episode(self):
        agent = ConstantAgent(action=0)
        seeds = make_seeds(42, 3)

        summary = evaluate_agent(agent, seeds, max_steps=20)

        self.assertEqual(len(summary.episodes), 3)
        self.assertEqual(agent.reset_count, 3)
        self.assertEqual(summary.steps, 20)


if __name__ == "__main__":
    unittest.main()