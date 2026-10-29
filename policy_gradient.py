"""Small, reproducible REINFORCE experiments for Homework 1.

J(theta) = E[sum_t gamma**t r_t]. The unbiased reward-to-go estimator is
sum_t gamma**t (G_t - b) grad log pi(a_t|s_t); the plain variant uses b=0.
Pong uses only ALE actions 2/3, with binary consecutive-frame differences.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, inputs: int, hidden: int = 32):
        super().__init__()
        activation = nn.ReLU() if inputs == 6400 else nn.Tanh()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), activation, nn.Linear(hidden, 2))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


@dataclass(frozen=True)
class EmaBaseline:
    value: float = 0.0
    decay: float = 0.9

    def updated(self, returns: np.ndarray) -> 'EmaBaseline':
        value = self.decay * self.value + (1 - self.decay) * float(np.mean(returns))
        return replace(self, value=value)


def discounted_returns(rewards, gamma: float) -> np.ndarray:
    if not 0 <= gamma <= 1:
        raise ValueError('gamma must be between 0 and 1')
    result = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def policy_loss(log_prob, returns, gamma: float, baseline: float = 0.0):
    powers = gamma ** torch.arange(len(returns), dtype=returns.dtype)
    return -(log_prob * powers * (returns - baseline)).sum()


def preprocess(image: np.ndarray) -> np.ndarray:
    # Copy prevents altering the environment observation through NumPy views.
    cropped = image[35:195:2, ::2, 0].copy()
    foreground = (cropped != 144) & (cropped != 109) & (cropped != 0)
    return foreground.astype(np.float32)


def make_env(task: str):
    if task == 'cartpole':
        return gym.make('CartPole-v1')
    if task != 'pong':
        raise ValueError('task must be cartpole or pong')
    import ale_py
    gym.register_envs(ale_py)
    return gym.make('ALE/Pong-v5', frameskip=4, repeat_action_probability=0.25)


def episode(actor, env, task, seed, rng, keep_states=True, random_policy=False):
    observation, _ = env.reset(seed=seed)
    previous = None
    states, actions, rewards, trace = [], [], [], []
    terminated = truncated = False
    with torch.no_grad():
        while not (terminated or truncated):
            if task == 'pong':
                current = preprocess(observation)
                state = (current - previous if previous is not None else np.zeros_like(current)).ravel()
                previous = current
            else:
                state = np.asarray(observation, dtype=np.float32)
            probabilities = np.array([0.5, 0.5]) if random_policy else actor(torch.from_numpy(state)).softmax(-1).numpy()
            action_index = int(rng.random() >= float(probabilities[0]))
            action = action_index + 2 if task == 'pong' else action_index
            observation, reward, terminated, truncated, _ = env.step(action)
            if keep_states:
                states.append(state)
                actions.append(action_index)
            rewards.append(float(reward))
            if len(trace) < 10:
                trace.append({'step': len(rewards)-1, 'state_norm': float(np.linalg.norm(state)),
                              'state': state.tolist() if task == 'cartpole' else None,
                              'changed_pixels': int(np.count_nonzero(state)) if task == 'pong' else None,
                              'p_action_0': float(probabilities[0]), 'action': action,
                              'reward': float(reward), 'terminated': bool(terminated),
                              'truncated': bool(truncated)})
    return {'states': states, 'actions': actions, 'rewards': rewards, 'trace': trace,
            'terminated': bool(terminated), 'truncated': bool(truncated)}


def log_probabilities(actor, trajectory):
    states = torch.from_numpy(np.asarray(trajectory['states'], dtype=np.float32))
    actions = torch.tensor(trajectory['actions'], dtype=torch.int64)
    return actor(states).log_softmax(-1).gather(1, actions[:, None]).squeeze(1)


def gradient_pair(actor, trajectory, gamma, baseline):
    log_prob = log_probabilities(actor, trajectory)
    returns = torch.from_numpy(discounted_returns(trajectory['rewards'], gamma))
    gradients = []
    for index, value in enumerate((0.0, baseline)):
        loss = policy_loss(log_prob, returns, gamma, value)
        tensors = torch.autograd.grad(loss, actor.parameters(), retain_graph=index == 0)
        gradients.append(torch.cat([part.ravel() for part in tensors]).numpy())
    return gradients


def gradient_summary(pairs, baseline):
    result = {'episodes': len(pairs), 'constant_baseline': baseline}
    for index, label in enumerate(('plain', 'baseline')):
        values = np.stack([pair[index] for pair in pairs]).astype(np.float64)
        mean = values.mean(axis=0)
        result[label] = {'mean_squared_norm': float(np.mean(np.sum(values**2, axis=1))),
                         'variance_trace': float(np.mean(np.sum((values-mean)**2, axis=1))),
                         'squared_mean_norm': float(np.sum(mean**2))}
    result['variance_ratio_baseline_over_plain'] = result['baseline']['variance_trace'] / max(result['plain']['variance_trace'], 1e-30)
    return result


def evaluate(actor, task, episodes=500, seed=20000, diagnostics=None, random_policy=False):
    env = make_env(task)
    rng = np.random.default_rng(seed)
    actor.eval()
    rows, pairs = [], []
    gamma = 0.95 if task == 'cartpole' else 0.99
    try:
        for index in range(episodes):
            save = diagnostics is not None and index < 20
            trajectory = episode(actor, env, task, seed+index, rng, save, random_policy)
            rewards = trajectory['rewards']
            rows.append({'episode': index+1, 'seed': seed+index, 'reward': sum(rewards),
                         'discounted_reward': float(discounted_returns(rewards, gamma)[0]),
                         'length': len(rewards), 'terminated': trajectory['terminated'],
                         'truncated': trajectory['truncated']})
            if save:
                pairs.append(gradient_pair(actor, trajectory, gamma, diagnostics['baseline']))
                if index == 0:
                    returns = discounted_returns(rewards, gamma)
                    diagnostics['first10_steps'] = [row | {
                        'return_to_go': float(returns[row['step']]),
                        'gamma_to_t': gamma ** row['step'],
                        'baseline_candidate': diagnostics['baseline'],
                    } for row in trajectory['trace']]
            if (index+1) % 100 == 0:
                print(f'{task} evaluate {index+1}/{episodes}', flush=True)
    finally:
        env.close()
    if diagnostics is not None:
        diagnostics['gradient_variance'] = gradient_summary(pairs, diagnostics['baseline'])
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reward_statistics(rows):
    rewards = np.array([row['reward'] for row in rows])
    return {'episodes': len(rows), 'mean': float(rewards.mean()),
            'std_population': float(rewards.std(ddof=0)), 'min': float(rewards.min()),
            'max': float(rewards.max()), 'mean_length': float(np.mean([r['length'] for r in rows])),
            'terminated': sum(row['terminated'] for row in rows),
            'truncated': sum(row['truncated'] for row in rows)}


def save_plots(output, task, use_baseline, training, evaluation):
    title = f'{"CartPole-v1" if task == "cartpole" else "ALE/Pong-v5"}: {"EMA baseline" if use_baseline else "plain REINFORCE"}'
    rewards = np.array([row['reward'] for row in training])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(np.arange(1, len(rewards)+1), rewards, color='#90aac4', alpha=0.6, linewidth=0.7, label='Episode reward')
    if len(rewards) >= 100:
        average = np.convolve(rewards, np.ones(100)/100, mode='valid')
        ax.plot(np.arange(100, len(rewards)+1), average, color='#b34b37', label='100-episode moving average')
    ax.set(xlabel='Training episode', ylabel='Undiscounted total reward', title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output/'training.png', dpi=160)
    plt.close(fig)
    values = np.array([row['reward'] for row in evaluation])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(-21.5, 22.5, 1) if task == 'pong' else 25
    ax.hist(values, bins=bins, color='#4a749b', edgecolor='white')
    ax.set(xlabel='Undiscounted total reward', ylabel='Number of episodes',
           title=f'{title}\n{len(values)} evaluation episodes: mean={values.mean():.2f}, SD={values.std():.2f}')
    fig.tight_layout()
    fig.savefig(output/'evaluation.png', dpi=160)
    plt.close(fig)


def train(args):
    if args.episodes < 1 or args.eval_episodes < 1 or args.learning_rate <= 0:
        raise ValueError('episode counts and learning rate must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gamma = 0.95 if args.task == 'cartpole' else 0.99
    actor = Actor(4 if args.task == 'cartpole' else 6400, args.hidden)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    baseline = EmaBaseline()
    env = make_env(args.task)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started, rows = time.perf_counter(), []
    try:
        for index in range(args.episodes):
            trajectory = episode(actor, env, args.task, args.seed+index, rng)
            returns = discounted_returns(trajectory['rewards'], gamma)
            b = baseline.value if args.baseline else 0.0
            loss = policy_loss(log_probabilities(actor, trajectory), torch.from_numpy(returns), gamma, b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            baseline = baseline.updated(returns)
            rows.append({'episode': index+1, 'seed': args.seed+index,
                         'reward': sum(trajectory['rewards']), 'discounted_reward': float(returns[0]),
                         'length': len(returns), 'baseline_used': b, 'loss': float(loss.detach()),
                         'terminated': trajectory['terminated'], 'truncated': trajectory['truncated']})
            if (index+1) % 25 == 0:
                recent = np.mean([r['reward'] for r in rows[-100:]])
                print(f'{args.task} baseline={args.baseline} train {index+1}/{args.episodes} recent={recent:.2f} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    finally:
        env.close()
    training_seconds = time.perf_counter()-started
    config = vars(args) | {'gamma': gamma, 'environment': 'CartPole-v1' if args.task == 'cartpole' else 'ALE/Pong-v5',
                           'baseline_decay': baseline.decay, 'final_baseline': baseline.value,
                           'objective': 'E[sum_t gamma**t r_t]', 'return_standardization': False,
                           'pong_frame_difference': args.task == 'pong', 'pong_actions': [2, 3] if args.task == 'pong' else None,
                           'pong_frameskip': 4 if args.task == 'pong' else None,
                           'pong_sticky_action_probability': 0.25 if args.task == 'pong' else None}
    torch.save({'state_dict': actor.state_dict(), 'config': config}, output/'policy.pt')
    write_csv(output/'training.csv', rows)
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    diagnostics = {'baseline': baseline.value}
    evaluation = evaluate(actor, args.task, args.eval_episodes, args.eval_seed, diagnostics)
    frozen = all(torch.equal(value, before[key]) for key, value in actor.state_dict().items())
    write_csv(output/'evaluation.csv', evaluation)
    metrics = {'config': config, 'training': reward_statistics(rows) | {
        'first100_mean': float(np.mean([r['reward'] for r in rows[:100]])),
        'last100_mean': float(np.mean([r['reward'] for r in rows[-100:]])),
        'seconds': training_seconds}, 'evaluation': reward_statistics(evaluation) | {'parameters_unchanged': frozen},
        'gradient_variance': diagnostics['gradient_variance'], 'total_seconds': time.perf_counter()-started}
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (output/'first10_steps.json').write_text(json.dumps(diagnostics['first10_steps'], indent=2)+'\n')
    save_plots(output, args.task, args.baseline, rows, evaluation)
    print(json.dumps(metrics), flush=True)


def toy_bandit_validation():
    # P(a=1)=0.3, r(a)=a. Enumerate the two actions exactly, no simulation error.
    probability = np.array([0.7, 0.3])
    score = np.array([-0.3, 0.7])
    rewards = np.array([0.0, 1.0])
    gradients = (score*rewards, score*(rewards-0.3))
    means = [float(np.sum(probability*g)) for g in gradients]
    variances = [float(np.sum(probability*(g-m)**2)) for g, m in zip(gradients, means)]
    return {'analytic_gradient': 0.21, 'plain_expected_gradient': means[0],
            'baseline_expected_gradient': means[1], 'plain_gradient_variance': variances[0],
            'baseline_gradient_variance': variances[1], 'variance_reduction_percent': 100*(1-variances[1]/variances[0])}


def validate(output):
    torch.set_num_threads(1)
    result = {'bandit_exact_enumeration': toy_bandit_validation(),
              'discounted_returns': {'rewards': [1, 2, 3], 'gamma': 0.5,
                                     'computed': discounted_returns([1, 2, 3], 0.5).tolist(),
                                     'expected': [2.75, 3.5, 3.0]},
              'random_policy': {}}
    for task in ('cartpole', 'pong'):
        rows = evaluate(Actor(4 if task == 'cartpole' else 6400), task, 50, 40000, random_policy=True)
        result['random_policy'][task] = reward_statistics(rows) | {'seed_start': 40000, 'uniform_actions': True}
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['cartpole', 'pong'], default='cartpole')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--episodes', type=int, default=800)
    parser.add_argument('--eval-episodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--eval-seed', type=int, default=20000)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--output', default='results/cartpole_plain')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    validate(args.output) if args.validate else train(args)


if __name__ == '__main__':
    main()
