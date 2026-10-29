"""Small analytic and exhaustive checks of the shuttle control problem."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shuttle import Parameters, build_model, finite_horizon, policy_iteration, value_iteration, run


class ShuttleTests(unittest.TestCase):
    def test_short_deterministic_horizon_matches_exhaustive_action_sequences(self):
        params = Parameters(capacity=3, max_queue=8, dispatch_cost=5,
                            holding_cost=1, gamma=0.8, arrivals=(2,))
        values, _ = finite_horizon(build_model(params), horizon=4)
        for initial in range(9):
            candidates = []
            for actions in itertools.product((0, 1), repeat=4):
                state, total = initial, 0.0
                for t, action in enumerate(actions):
                    remaining = max(0, state - 3 * action)
                    total += 0.8 ** t * (5 * action + remaining)
                    state = min(8, remaining + 2)
                candidates.append(total)
            self.assertAlmostEqual(values[initial], min(candidates), places=10)

    def test_zero_discount_is_greedy_and_never_dispatches(self):
        values, policy, _ = value_iteration(build_model(Parameters(gamma=0)))
        np.testing.assert_allclose(values, 2 * np.arange(201))
        np.testing.assert_array_equal(policy, np.zeros(201, dtype=int))

    def test_free_dispatch_serves_every_nonempty_queue(self):
        _, policy, _ = value_iteration(build_model(Parameters(dispatch_cost=0)))
        np.testing.assert_array_equal(policy, np.r_[0, np.ones(200, dtype=int)])

    def test_zero_holding_cost_makes_waiting_free(self):
        values, policy, _ = policy_iteration(build_model(Parameters(holding_cost=0)))
        np.testing.assert_array_equal(values, np.zeros(201))
        np.testing.assert_array_equal(policy, np.zeros(201, dtype=int))

    def test_no_arrivals_matches_closed_form_for_one_shuttle(self):
        model = build_model(Parameters(max_queue=15, arrivals=(0,)))
        values, policy, _ = value_iteration(model)
        expected = np.minimum(2 * np.arange(16) / 0.05, 100)
        np.testing.assert_allclose(values, expected, atol=2e-8)
        np.testing.assert_array_equal(policy, np.r_[np.zeros(3), np.ones(13)])

    def test_boundary_transitions_and_t500_agreement(self):
        model = build_model()
        np.testing.assert_array_equal(model.successors[200, 0], [200] * 5)
        np.testing.assert_array_equal(model.successors[200, 1], np.arange(186, 191))
        finite, _ = finite_horizon(model, 500)
        vi, vi_policy, _ = value_iteration(model)
        pi, pi_policy, _ = policy_iteration(model)
        np.testing.assert_allclose(finite, pi, atol=1e-7)
        np.testing.assert_allclose(vi, pi, atol=2e-8)
        np.testing.assert_array_equal(vi_policy, pi_policy)

    def test_complete_experiment_exports_consistent_numerical_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = run(output)
            saved = json.loads((output / 'summary.json').read_text())
            self.assertEqual(saved, json.loads(json.dumps(summary)))
            self.assertEqual(summary['dispatch_threshold'], 13)
            self.assertEqual(summary['last_period_dispatch_states'], 0)
            self.assertLess(summary['max_abs_difference_finite500_pi'], 1e-7)
            self.assertEqual(summary['controlled_experiments']['free_dispatch']['dispatch_states'], 200)
            self.assertEqual(summary['controlled_experiments']['free_holding']['max_abs_value'], 0)
            rows = np.genfromtxt(output / 'first_10_steps.csv', delimiter=',', names=True)
            np.testing.assert_array_equal(rows['next_state'][:-1], rows['state'][1:])
            np.testing.assert_allclose(rows['J'], np.minimum(rows['Q_wait'], rows['Q_dispatch']))
            for filename in ['finite_horizon.png', 'value_iteration.png', 'optimal_policy.png']:
                self.assertGreater((output / filename).stat().st_size, 1000)

    def test_zero_horizon_and_invalid_parameters(self):
        values, policies = finite_horizon(build_model(), 0)
        np.testing.assert_array_equal(values, np.zeros(201))
        self.assertEqual(policies.shape, (0, 201))
        for params in [Parameters(gamma=1), Parameters(capacity=0),
                       Parameters(arrivals=()), Parameters(arrivals=(-1,)),
                       Parameters(holding_cost=-1)]:
            with self.assertRaises(ValueError):
                build_model(params)


if __name__ == '__main__':
    unittest.main()
