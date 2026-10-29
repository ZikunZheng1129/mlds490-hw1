"""Finite enumeration, value iteration, and policy iteration for HW1.

Queue observed -> optional dispatch -> holding cost -> arrivals. Arrivals that
would exceed 200 are blocked, with no extra penalty. All values are COSTS J;
the reward value in the lecture's maximization convention is V = -J.
"""
from dataclasses import asdict, dataclass, replace
import argparse
import csv
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Parameters:
    capacity: int = 15
    max_queue: int = 200
    dispatch_cost: float = 100.0
    holding_cost: float = 2.0
    gamma: float = 0.95
    arrivals: tuple = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Model:
    parameters: Parameters
    costs: np.ndarray
    successors: np.ndarray


def build_model(parameters=Parameters()):
    """Enumerate 201 states, two actions, and equally likely arrival counts."""
    p = parameters
    if not isinstance(p.capacity, int) or p.capacity < 1:
        raise ValueError('capacity must be a positive integer')
    if not isinstance(p.max_queue, int) or p.max_queue < 0:
        raise ValueError('max_queue must be a nonnegative integer')
    if not np.isfinite(p.gamma) or not 0 <= p.gamma < 1:
        raise ValueError('gamma must satisfy 0 <= gamma < 1')
    if not all(np.isfinite(c) and c >= 0 for c in (p.dispatch_cost, p.holding_cost)):
        raise ValueError('costs must be finite and nonnegative')
    if not p.arrivals or any(not isinstance(a, int) or a < 0 for a in p.arrivals):
        raise ValueError('arrivals must contain nonnegative integers')
    states = np.arange(p.max_queue + 1)[:, None]
    actions = np.arange(2)[None, :]
    remaining = np.maximum(states - p.capacity * actions, 0)
    costs = p.dispatch_cost * actions + p.holding_cost * remaining
    successors = np.minimum(remaining[:, :, None] + np.array(p.arrivals), p.max_queue)
    return Model(p, costs, successors)


def action_costs(model, values):
    """Bellman right-hand sides for actions 0 (wait) and 1 (dispatch)."""
    return model.costs + model.parameters.gamma * values[model.successors].mean(axis=2)


def finite_horizon(model, horizon=500):
    """Backward induction enumerating each state/action; J_T = 0, t=T-1,...,0."""
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError('horizon must be a nonnegative integer')
    values = np.zeros(model.costs.shape[0])
    reverse_policies = []
    for _ in range(horizon):
        q = action_costs(model, values)
        reverse_policies.append(q.argmin(axis=1))
        values = q.min(axis=1)
    policies = np.array(reverse_policies[::-1], dtype=int).reshape(horizon, len(values))
    return values, policies


def value_iteration(model, tolerance=1e-10, max_iterations=10000):
    """Stop at a small sup-norm change; report the independent Bellman residual."""
    if tolerance <= 0 or not np.isfinite(tolerance) or max_iterations < 1:
        raise ValueError('positive finite tolerance and iteration limit are required')
    values = np.zeros(model.costs.shape[0])
    for iteration in range(1, max_iterations + 1):
        updated = action_costs(model, values).min(axis=1)
        difference = float(np.max(np.abs(updated - values)))
        values = updated
        if difference < tolerance:
            q = action_costs(model, values)
            residual = float(np.max(np.abs(q.min(axis=1) - values)))
            return values, q.argmin(axis=1), {'iterations': iteration, 'residual': residual}
    raise RuntimeError('value iteration did not converge')


def evaluate_policy(model, policy):
    """Solve (I-gamma P_pi)J_pi=c_pi; aggregate arrivals clipped at the cap."""
    states = np.arange(model.costs.shape[0])
    next_states = model.successors[states, policy]
    transition = (next_states[:, :, None] == states).mean(axis=1)
    matrix = np.eye(len(states)) - model.parameters.gamma * transition
    return np.linalg.solve(matrix, model.costs[states, policy])


def policy_iteration(model, max_iterations=100):
    """Exact policy evaluation followed by greedy improvement; ties choose wait."""
    policy = np.zeros(model.costs.shape[0], dtype=int)
    for iteration in range(1, max_iterations + 1):
        values = evaluate_policy(model, policy)
        q = action_costs(model, values)
        improved = q.argmin(axis=1)
        if np.array_equal(improved, policy):
            residual = float(np.max(np.abs(q.min(axis=1) - values)))
            return values, policy, {'iterations': iteration, 'residual': residual}
        policy = improved
    raise RuntimeError('policy iteration did not converge')


def exhaustive_deterministic_error():
    """Independent 2^4 path enumeration checks a small deterministic instance."""
    from itertools import product
    p = Parameters(capacity=3, max_queue=8, dispatch_cost=5,
                   holding_cost=1, gamma=0.8, arrivals=(2,))
    values, _ = finite_horizon(build_model(p), 4)
    errors = []
    for initial in range(9):
        totals = []
        for actions in product((0, 1), repeat=4):
            state, total = initial, 0.0
            for t, action in enumerate(actions):
                remaining = max(0, state - 3 * action)
                total += 0.8 ** t * (5 * action + remaining)
                state = min(8, remaining + 2)
            totals.append(total)
        errors.append(abs(values[initial] - min(totals)))
    return float(max(errors))


def controlled_experiments(parameters):
    """Parameter interventions with analytic policy/value predictions."""
    zero_gamma = build_model(replace(parameters, gamma=0))
    greedy_values, greedy_policy, _ = value_iteration(zero_gamma)
    free_model = build_model(replace(parameters, dispatch_cost=0))
    free_values, free_policy, _ = value_iteration(free_model)
    free_holding = build_model(replace(parameters, holding_cost=0))
    wait_values, wait_policy, _ = policy_iteration(free_holding)
    deterministic = build_model(replace(parameters, max_queue=15, arrivals=(0,)))
    deterministic_values, deterministic_policy, _ = value_iteration(deterministic)
    analytic = np.minimum(2 * np.arange(16) / (1 - parameters.gamma), 100)
    return {
        'gamma_zero': {'dispatch_states': int(greedy_policy.sum()),
                       'max_error_vs_2n': float(np.max(np.abs(greedy_values - 2 * np.arange(201))))},
        'free_dispatch': {'dispatch_states': int(free_policy.sum()),
                          'nonempty_states': len(free_policy) - 1, 'J0': float(free_values[0])},
        'free_holding': {'dispatch_states': int(wait_policy.sum()),
                         'max_abs_value': float(np.max(np.abs(wait_values)))},
        'no_arrivals': {'dispatch_threshold': int(np.flatnonzero(deterministic_policy)[0]),
                        'max_error_vs_min_40n_100': float(np.max(np.abs(deterministic_values - analytic)))},
        'exhaustive_4_step_2_actions_9_states_max_error': exhaustive_deterministic_error(),
    }


def trace_policy(model, values, policy, initial=0, seed=490, steps=10):
    """Seeded trace records observed costs as well as Bellman action costs."""
    rng = np.random.default_rng(seed)
    q = action_costs(model, values)
    state = initial
    rows = []
    for t in range(steps):
        action = int(policy[state])
        arrival = int(rng.choice(model.parameters.arrivals))
        remaining = max(0, state - model.parameters.capacity * action)
        next_state = min(model.parameters.max_queue, remaining + arrival)
        rows.append({'t': t, 'state': state, 'action': action, 'arrival': arrival,
                     'remaining': remaining, 'cost': float(model.costs[state, action]),
                     'next_state': next_state, 'J': float(values[state]),
                     'Q_wait': float(q[state, 0]), 'Q_dispatch': float(q[state, 1])})
        state = next_state
    return rows


def save_plots(output, finite, infinite, policy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states = np.arange(len(infinite))
    for name, values, title in [
        ('finite_horizon.png', finite, 'Finite horizon enumeration (T = 500)'),
        ('value_iteration.png', infinite, 'Infinite horizon value iteration')]:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(states, values, linewidth=2)
        ax.set(xlabel='Customers waiting at decision time',
               ylabel='Optimal discounted cost J', title=title, xlim=(0, 200))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.step(states, policy, where='post', linewidth=2)
    threshold = int(np.flatnonzero(policy)[0])
    ax.axvline(threshold, color='gray', linestyle='--', label=f'Dispatch from n = {threshold}')
    ax.set(xlabel='Customers waiting at decision time', ylabel='Optimal action',
           title='Policy iteration', xlim=(0, 200), ylim=(-0.1, 1.1), yticks=[0, 1])
    ax.set_yticklabels(['Wait', 'Dispatch'])
    ax.legend(loc='lower right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / 'optimal_policy.png', dpi=160)
    plt.close(fig)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    model = build_model()
    finite, finite_policies = finite_horizon(model)
    vi, vi_policy, vi_info = value_iteration(model)
    pi, pi_policy, pi_info = policy_iteration(model)
    states = np.arange(201)
    np.savetxt(output / 'values_and_policy.csv',
               np.column_stack([states, finite, vi, pi, pi_policy]), delimiter=',',
               header='queue,J_finite_500,J_value_iteration,J_policy_iteration,action', comments='')
    trace = trace_policy(model, pi, pi_policy)
    with (output / 'first_10_steps.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    convergence = [{'horizon': horizon, 'max_abs_error_vs_infinite': float(
        np.max(np.abs(finite_horizon(model, horizon)[0] - pi)))}
        for horizon in [1, 10, 50, 100, 200, 500]]
    summary = {
        'parameters': asdict(model.parameters), 'value_convention': 'discounted cost J; reward value V = -J',
        'terminal_cost': 0, 'overflow': 'blocked at cap, no extra penalty',
        'value_iteration': vi_info, 'policy_iteration': pi_info,
        'dispatch_threshold': int(np.flatnonzero(pi_policy)[0]),
        'policy_mismatches_vi_pi': int(np.sum(vi_policy != pi_policy)),
        'policy_mismatches_finite_t0_pi': int(np.sum(finite_policies[0] != pi_policy)),
        'last_period_dispatch_states': int(np.sum(finite_policies[-1])),
        'max_abs_difference_vi_pi': float(np.max(np.abs(vi - pi))),
        'max_abs_difference_finite500_pi': float(np.max(np.abs(finite - pi))),
        'vi_residual_error_bound': vi_info['residual'] / (1 - model.parameters.gamma),
        'finite500_tail_bound': 470 * model.parameters.gamma ** 500 / (1 - model.parameters.gamma),
        'selected_values': {str(n): float(pi[n]) for n in [0, 1, 12, 13, 15, 200]},
        'horizon_convergence': convergence,
        'controlled_experiments': controlled_experiments(model.parameters),
        'boundary': {'wait_at_200_next_states': model.successors[200, 0].tolist(),
                     'dispatch_at_200_next_states': model.successors[200, 1].tolist()},
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    save_plots(output, finite, vi, pi_policy)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results' / 'shuttle')
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
