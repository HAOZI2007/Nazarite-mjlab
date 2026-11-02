"""Contract tests runnable with the standard library unittest runner."""

import unittest

import numpy as np
from Sim2Real.xz1_him.runtime import (
    Controller,
    ObservationBuilder,
    RobotState,
    load_config,
)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.state = RobotState(
            self.config["default_q"].copy(),
            np.zeros(12),
            np.zeros(3),
            np.array([1, 0, 0, 0]),
        )

    def test_frame_order_phase_and_reset(self):
        builder = ObservationBuilder(self.config)
        obs = builder.build(self.state, [0.2, 0, 0], np.zeros(12)).reshape(6, 47)
        np.testing.assert_array_equal(obs[0], obs[5])
        np.testing.assert_array_equal(obs[0, 3:6], [0, 0, -1])
        np.testing.assert_array_equal(obs[0, 9:11], [0, 1])
        np.testing.assert_array_equal(obs[0, 11:35], 0)
        builder.advance()
        self.state.joint_pos[0] += 0.1
        next_obs = builder.build(self.state, [0.2, 0, 0], np.ones(12)).reshape(6, 47)
        np.testing.assert_array_equal(next_obs[1], obs[0])
        np.testing.assert_allclose(next_obs[0, 11], 0.1)
        np.testing.assert_array_equal(next_obs[0, 35:], 1)
        self.assertGreater(next_obs[0, 9], 0)
        builder.reset()
        self.assertFalse(builder.initialized)
        self.assertEqual(builder.phase, 0)

    def test_gravity_and_invalid_state(self):
        builder = ObservationBuilder(self.config)
        self.state.quaternion_wxyz = np.array([2**-0.5, 2**-0.5, 0, 0])
        obs = builder.build(self.state, np.zeros(3), np.zeros(12))
        np.testing.assert_allclose(obs[3:6], [0, -1, 0], atol=2e-7)
        self.state.angular_velocity[0] = np.nan
        with self.assertRaises(ValueError):
            builder.build(self.state, np.zeros(3), np.zeros(12))

    def test_offset_and_rejected_target(self):
        class Policy:
            action = np.zeros(12, dtype=np.float32)

            def step(self, obs):
                return self.action.copy()

        policy = Policy()
        controller = Controller(policy, self.config)
        target, _ = controller.step(self.state, np.zeros(3))
        np.testing.assert_array_equal(target.position, self.config["action_offset"])
        policy.action[:] = 10
        with self.assertRaises(ValueError):
            controller.step(self.state, np.zeros(3))
        np.testing.assert_array_equal(controller.last_action, 0)


if __name__ == "__main__":
    unittest.main()
