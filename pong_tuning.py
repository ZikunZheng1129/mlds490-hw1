"""Pong REINFORCE experiments; the report uses the ishan configuration.

Independent implementations of settings inspected in:
https://github.com/yipinggao0109/mlds-490-hw1
https://github.com/ishan-monie/advanced-ml-hw1
These variants omit the outer gamma**t factor; Ishan's also resets returns at points.
They are different estimators from the original start-state discounted objective.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from policy_gradient import (
    Actor, discounted_returns, episode, make_env, preprocess,
    reward_statistics, save_plots, write_csv,
)


@dataclass(frozen=True)
class Config:
    recipe: str
    episodes: int
    seed: int
    hidden: int
    batch_episodes: int
    loss_divisor: str
    reset_at_points: bool = False
    max_grad_norm: float | None = None
    chunk_size: int = 0
    eval_episodes: int = 500
    gamma: float = .99
    learning_rate: float = .001
    outer_time_discount: bool = False


def recipe_config(name):
    if name == 'uniform-only':
        return Config(name, 1000, 7, 200, 1, 'sum')
    if name == 'yiping':
        return Config(name, 5000, 490, 200, 10, 'steps', max_grad_norm=5., chunk_size=1024)
    if name == 'ishan':
        return Config(name, 3000, 1, 256, 4, 'episodes', reset_at_points=True)
    raise ValueError('unknown recipe')


def validate_config(config):
    if min(config.episodes, config.eval_episodes, config.hidden, config.batch_episodes) < 1:
        raise ValueError('episode counts, hidden units and batch size must be positive')
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0 or not 0 <= config.gamma <= 1:
        raise ValueError('invalid learning rate or discount')
    if config.loss_divisor not in ('sum', 'steps', 'episodes') or config.chunk_size < 0:
        raise ValueError('invalid loss scaling or chunk size')


class BernoulliActor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6400, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, states):
        return self.network(states).squeeze(-1)


def make_actor(config):
    return BernoulliActor(config.hidden) if config.recipe == 'ishan' else Actor(6400, config.hidden)


def returns_to_go(rewards, gamma=.99, reset_at_points=False):
    if not reset_at_points:
        return discounted_returns(rewards, gamma)
    values = np.empty(len(rewards), dtype=np.float32)
    running = 0.
    for index in range(len(rewards)-1, -1, -1):
        running = 0. if rewards[index] != 0 else running
        running = float(rewards[index]) + gamma * running
        values[index] = running
    return values


def collect(actor, env, config, number, rng, evaluating=False):
    if config.recipe == 'uniform-only':
        seed = (20000 if evaluating else config.seed) + number - 1
        game = episode(actor, env, 'pong', seed, rng)
        return game | {'startup_fire': 0, 'startup_reward': 0.}
    seed = (config.seed + (100000 if evaluating else 0) + number
            if config.recipe == 'yiping' else None)
    observation, _ = env.reset(seed=seed)
    startup_fire, startup_reward = 0, 0.
    if config.recipe == 'yiping':
        meanings = env.unwrapped.get_action_meanings()
        if 'FIRE' in meanings:
            observation, startup_reward, terminated, truncated, _ = env.step(meanings.index('FIRE'))
            startup_fire = 1
            if terminated or truncated:
                observation, _ = env.reset(seed=seed)
    previous = None
    states, actions, rewards = [], [], []
    terminated = truncated = False
    with torch.no_grad():
        while not (terminated or truncated):
            current = preprocess(observation).ravel()
            first = current if config.recipe == 'yiping' else np.zeros_like(current)
            state = current - previous if previous is not None else first
            previous = current
            inputs = torch.from_numpy(state)
            if config.recipe == 'ishan':
                action = int(Bernoulli(logits=actor(inputs)).sample().item())
                ale_action = 2 if action == 1 else 3
            else:
                action = int(Categorical(logits=actor(inputs[None])).sample().item())
                ale_action = action + 2
            observation, reward, terminated, truncated, _ = env.step(ale_action)
            states.append(state.astype(np.int8))
            actions.append(action)
            rewards.append(float(reward))
    return {'states': states, 'actions': actions, 'rewards': rewards,
            'terminated': bool(terminated), 'truncated': bool(truncated),
            'startup_fire': startup_fire, 'startup_reward': float(startup_reward)}


def update(actor, optimizer, games, config):
    if not games or any(len(game['rewards']) == 0 for game in games):
        raise ValueError('an update requires nonempty complete episodes')
    states = np.concatenate([np.asarray(g['states'], dtype=np.int8) for g in games])
    actions = np.concatenate([np.asarray(g['actions'], dtype=np.int64) for g in games])
    returns = np.concatenate([returns_to_go(g['rewards'], config.gamma, config.reset_at_points) for g in games])
    n_steps = len(returns)
    divisor = {'sum': 1, 'steps': n_steps, 'episodes': len(games)}[config.loss_divisor]
    size = config.chunk_size or n_steps
    optimizer.zero_grad()
    total_loss = 0.
    for start in range(0, n_steps, size):
        inputs = torch.as_tensor(states[start:start+size], dtype=torch.float32)
        selected = torch.as_tensor(actions[start:start+size])
        logits = actor(inputs)
        if config.recipe == 'ishan':
            logp = Bernoulli(logits=logits).log_prob(selected.float())
        elif config.recipe == 'yiping':
            logp = Categorical(logits=logits).log_prob(selected)
        else:
            logp = logits.log_softmax(-1).gather(1, selected[:, None]).squeeze(1)
        loss = -(logp * torch.from_numpy(returns[start:start+size])).sum() / divisor
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite policy loss')
        loss.backward()
        total_loss += float(loss.detach())
    norm = None
    if config.max_grad_norm is not None:
        norm = float(nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm))
    optimizer.step()
    return {'loss': total_loss, 'gradient_norm_before_clipping': norm, 'batch_steps': n_steps}


def snapshot(path, actor, optimizer, config, completed, updates, rng):
    torch.save({'state_dict': actor.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'config': asdict(config), 'completed_episodes': completed, 'optimizer_updates': updates,
                'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': rng.bit_generator.state,
                'exact_resume_supported': False}, path)


def episode_row(game, number):
    rewards = game['rewards']
    return {'episode': number, 'reward': sum(rewards), 'length': len(rewards),
            'discounted_reward': float(discounted_returns(rewards, .99)[0]),
            'terminated': game['terminated'], 'truncated': game['truncated'],
            'startup_fire': game['startup_fire'], 'startup_reward': game['startup_reward']}


def evaluate(actor, config):
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    actor.eval()
    env = make_env('pong')
    if config.recipe == 'ishan':
        env.reset(seed=12345)
        torch.manual_seed(12345)
    elif config.recipe == 'yiping':
        env.action_space.seed(config.seed)
    rng = np.random.default_rng(20000)
    rows = []
    try:
        for number in range(1, config.eval_episodes+1):
            rows.append(episode_row(collect(actor, env, config, number, rng, True), number))
            if number % 100 == 0:
                print(f'{config.recipe} evaluate {number}/{config.eval_episodes}', flush=True)
    finally:
        env.close()
    frozen = all(torch.equal(before[key], value) for key, value in actor.state_dict().items())
    return rows, frozen


def training_batch(actor, env, config, first, rng):
    last = min(first + config.batch_episodes - 1, config.episodes)
    return [collect(actor, env, config, number, rng) for number in range(first, last+1)]


def configuration_record(config):
    peer = {'yiping': 'https://github.com/yipinggao0109/mlds-490-hw1',
            'ishan': 'https://github.com/ishan-monie/advanced-ml-hw1'}
    return asdict(config) | {'reference': peer.get(config.recipe),
        'estimator': 'discounted reward-to-go without outer gamma^t; point-segmented for ishan',
        'baseline': 'none', 'return_standardization': False, 'entropy_coefficient': 0.,
        'environment': 'ALE/Pong-v5', 'frameskip': 4, 'sticky_action_probability': .25,
        'policy_actions': [2, 3], 'startup_fire': config.recipe == 'yiping',
        'initial_frame': 'current' if config.recipe == 'yiping' else 'zeros',
        'sampling': 'numpy draw against policy probabilities' if config.recipe == 'uniform-only' else 'torch_distribution',
        'device': 'cpu', 'torch_threads': 1,
        'training_environment_seed': {'uniform-only': 'seed + episode - 1',
            'yiping': 'seed + episode', 'ishan': 'initialize once with seed; then unseeded resets'}[config.recipe],
        'evaluation_environment_seed': {'uniform-only': '20000 + episode - 1',
            'yiping': 'seed + 100000 + episode',
            'ishan': 'initialize once with 12345; then unseeded resets'}[config.recipe],
        'evaluation_action_rng': {'uniform-only': 'new NumPy generator seeded 20000',
            'yiping': 'continue training Torch RNG', 'ishan': 'reset Torch RNG to 12345'}[config.recipe],
        'state_storage': 'int8; exact for binary frame differences',
        'resume_note': 'Snapshots are for inspection; no exact resume is implemented.'}


def train(config, output):
    validate_config(config)
    output = Path(output)
    if output.exists() and any(p.name != 'run.log' for p in output.iterdir()):
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(configuration_record(config), indent=2)+'\n')
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    actor = make_actor(config)
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    env = make_env('pong')
    if config.recipe == 'ishan':
        env.reset(seed=config.seed)
    elif config.recipe == 'yiping':
        env.action_space.seed(config.seed)
    started, rows, updates = time.perf_counter(), [], 0
    try:
        with (output/'training.csv').open('w', newline='') as stream:
            writer = None
            for first in range(1, config.episodes+1, config.batch_episodes):
                actor.eval()
                games = training_batch(actor, env, config, first, rng)
                actor.train()
                info = update(actor, optimizer, games, config)
                updates += 1
                batch_rows = [episode_row(game, first+i) | info | {'update': updates} for i, game in enumerate(games)]
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(batch_rows[0]))
                    writer.writeheader()
                writer.writerows(batch_rows)
                stream.flush()
                rows.extend(batch_rows)
                completed = len(rows)
                if completed % 100 == 0 or completed == config.episodes:
                    recent = np.mean([r['reward'] for r in rows[-100:]])
                    print(f'{config.recipe} train {completed}/{config.episodes} mean100={recent:.2f} '
                          f'updates={updates} elapsed={time.perf_counter()-started:.1f}s', flush=True)
                if completed in (1000, 3000, config.episodes):
                    snapshot(output/f'checkpoint_{completed:05d}.pt', actor, optimizer, config, completed, updates, rng)
    finally:
        env.close()
    train_seconds = time.perf_counter()-started
    snapshot(output/'policy.pt', actor, optimizer, config, len(rows), updates, rng)
    evaluation, frozen = evaluate(actor, config)
    write_csv(output/'evaluation.csv', evaluation)
    metrics = {'config': configuration_record(config), 'optimizer_updates': updates,
               'training': reward_statistics(rows) | {'first100_mean': float(np.mean([r['reward'] for r in rows[:100]])),
                   'last100_mean': float(np.mean([r['reward'] for r in rows[-100:]])), 'seconds': train_seconds},
               'evaluation': reward_statistics(evaluation) | {'parameters_unchanged': frozen,
                   'wins': sum(r['reward'] > 0 for r in evaluation)},
               'training_startup_fire_actions': sum(r['startup_fire'] for r in rows),
               'evaluation_startup_fire_actions': sum(r['startup_fire'] for r in evaluation),
               'total_seconds': time.perf_counter()-started}
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    save_plots(output, 'pong', 'none', rows, evaluation)
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=['uniform-only', 'yiping', 'ishan'], required=True)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eval-episodes', type=int, default=500)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = recipe_config(args.recipe)
    config = replace(config, episodes=args.episodes if args.episodes is not None else config.episodes,
                     seed=args.seed if args.seed is not None else config.seed, eval_episodes=args.eval_episodes)
    train(config, args.output)


if __name__ == '__main__':
    main()
