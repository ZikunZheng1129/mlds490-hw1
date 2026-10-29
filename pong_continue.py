"""Warm-start the selected Pong policy; ALE state is not restored.

Actor, Adam state and training RNG are inherited. The new environment is seeded
separately; evaluation uses a fixed, separate seed and never updates the policy.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from pong_tuning import (
    Config, collect, configuration_record, episode_row, make_actor, make_env,
    reward_statistics, save_plots, snapshot, update, validate_config,
)

CHECKPOINT_EVERY = 250
LOG_EVERY = 50
EVALUATION_FIELDS = ('episode', 'reward', 'length', 'discounted_reward',
                     'terminated', 'truncated', 'startup_fire', 'startup_reward')


@dataclass(frozen=True)
class Parent:
    source: Path
    checkpoint: dict
    config: Config
    history: tuple
    fieldnames: tuple
    history_bytes: bytes
    source_sha256: str


def parse_history_row(row):
    integers = ('episode', 'length', 'startup_fire', 'batch_steps', 'update')
    floats = ('reward', 'discounted_reward', 'startup_reward', 'loss')
    if any(row[name] not in ('True', 'False') for name in ('terminated', 'truncated')):
        raise ValueError('invalid termination flag in parent history')
    parsed = ({name: int(row[name]) for name in integers}
              | {name: float(row[name]) for name in floats}
              | {name: row[name] == 'True' for name in ('terminated', 'truncated')}
              | {'gradient_norm_before_clipping':
                 float(row['gradient_norm_before_clipping']) if row['gradient_norm_before_clipping'] else None})
    if any(not np.isfinite(parsed[name]) for name in floats):
        raise ValueError('non-finite value in parent history')
    return parsed


def load_parent(source):
    source = Path(source)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    config = Config(**checkpoint['config'])
    validate_config(config)
    if config.recipe != 'ishan' or not config.reset_at_points or config.outer_time_discount:
        raise ValueError('continuation requires the selected point-return Pong configuration')
    raw = (source.parent/'training.csv').read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8')))
    history = tuple(parse_history_row(row) for row in reader)
    completed = checkpoint['completed_episodes']
    updates = checkpoint['optimizer_updates']
    if not history or len(history) != completed or config.episodes != completed:
        raise ValueError('parent history count does not match the completed checkpoint')
    if any(row['episode'] != i for i, row in enumerate(history, 1)):
        raise ValueError('parent history episode numbers are not continuous')
    if history[-1]['update'] != updates or any(not 1 <= row['update'] <= updates for row in history):
        raise ValueError('parent history update count does not match checkpoint')
    return Parent(source, checkpoint, config, history, tuple(reader.fieldnames), raw,
                  hashlib.sha256(source.read_bytes()).hexdigest())


def restore_training(parent, config, train_env_seed):
    actor = make_actor(config)
    actor.load_state_dict(parent.checkpoint['state_dict'])
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError('parent policy contains non-finite parameters')
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(parent.checkpoint['optimizer_state_dict'])
    for group in optimizer.param_groups:
        group['lr'] = config.learning_rate
    env = make_env('pong')
    try:
        env.reset(seed=train_env_seed)
        rng = np.random.default_rng()
        rng.bit_generator.state = parent.checkpoint['numpy_rng_state']
        torch.set_rng_state(parent.checkpoint['torch_rng_state'])
        return actor, optimizer, rng, env
    except Exception:
        env.close()
        raise


def run_record(parent, config, additional_episodes, train_env_seed, eval_seed):
    warm_start = {'source_checkpoint': str(parent.source), 'source_sha256': parent.source_sha256,
                  'inherited_episodes': len(parent.history),
                  'inherited_optimizer_updates': parent.checkpoint['optimizer_updates'],
                  'additional_episodes': additional_episodes, 'train_env_seed': train_env_seed,
                  'evaluation_seed': eval_seed, 'exact_resume': False,
                  'environment_state_restored': False, 'optimizer_state_restored': True,
                  'training_rng_restored': True}
    return configuration_record(config) | {
        'warm_start': warm_start,
        'training_environment_seed': f'fresh environment initialized with {train_env_seed}; then unseeded resets',
        'evaluation_environment_seed': f'fresh environment initialized with {eval_seed}; then unseeded resets',
        'evaluation_action_rng': f'Torch RNG initialized with {eval_seed}',
        'resume_note': 'Warm start with actor, Adam and training RNG; ALE state is not saved or restored.',
    }


def train_more(parent, actor, optimizer, rng, env, config, output):
    old_count = len(parent.history)
    updates = parent.checkpoint['optimizer_updates']
    rows, started = [], time.perf_counter()
    with (output/'training.csv').open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=parent.fieldnames)
        for first in range(old_count+1, config.episodes+1, config.batch_episodes):
            actor.eval()
            last = min(first+config.batch_episodes-1, config.episodes)
            games = [collect(actor, env, config, number, rng) for number in range(first, last+1)]
            actor.train()
            info = update(actor, optimizer, games, config)
            updates += 1
            batch = [episode_row(game, first+i) | info | {'update': updates}
                     for i, game in enumerate(games)]
            before = len(rows)
            rows = rows + batch
            writer.writerows(batch)
            stream.flush()
            completed = old_count + len(rows)
            if len(rows)//LOG_EVERY > before//LOG_EVERY or completed == config.episodes:
                recent = (list(parent.history) + rows)[-100:]
                mean = np.mean([row['reward'] for row in recent])
                print(f'continue train {completed}/{config.episodes} mean100={mean:.2f} '
                      f'updates={updates} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            if len(rows)//CHECKPOINT_EVERY > before//CHECKPOINT_EVERY or completed == config.episodes:
                snapshot(output/f'checkpoint_{completed:05d}.pt', actor, optimizer,
                         config, completed, updates, rng)
    return rows, updates, time.perf_counter()-started


def evaluate_final(actor, config, output, eval_seed):
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    actor.eval()
    env = make_env('pong')
    rows = []
    try:
        env.reset(seed=eval_seed)
        torch.manual_seed(eval_seed)
        rng = np.random.default_rng(eval_seed)
        with (output/'evaluation.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=EVALUATION_FIELDS)
            writer.writeheader()
            stream.flush()
            for number in range(1, config.eval_episodes+1):
                row = episode_row(collect(actor, env, config, number, rng, evaluating=True), number)
                rows = rows + [row]
                writer.writerow(row)
                stream.flush()
                if number % LOG_EVERY == 0 or number == config.eval_episodes:
                    mean = np.mean([row['reward'] for row in rows])
                    wins = sum(row['reward'] > 0 for row in rows)
                    print(f'continue evaluate {number}/{config.eval_episodes} '
                          f'mean={mean:.2f} wins={wins}/{number}', flush=True)
    finally:
        env.close()
    frozen = all(torch.equal(before[key], value) for key, value in actor.state_dict().items())
    return rows, frozen


def meets_target(evaluation):
    return bool(evaluation['episodes'] == 500 and evaluation['mean'] > 0
                and evaluation['wins'] > 250 and evaluation['parameters_unchanged']
                and evaluation['terminated'] == 500 and evaluation['truncated'] == 0)


def validate_options(additional_episodes, eval_episodes, train_env_seed, eval_seed, learning_rate):
    for value in (additional_episodes, eval_episodes):
        if not isinstance(value, int) or value < 1:
            raise ValueError('episode counts must be positive integers')
    for value in (train_env_seed, eval_seed):
        if not isinstance(value, int) or value < 0:
            raise ValueError('environment seeds must be nonnegative integers')
    if learning_rate is not None and (not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('learning rate must be finite and positive')


def continue_training(source, output, additional_episodes=1000, train_env_seed=70001,
                      eval_seed=54321, learning_rate=None, eval_episodes=500):
    validate_options(additional_episodes, eval_episodes, train_env_seed, eval_seed, learning_rate)
    output = Path(output)
    if output.exists() and any(p.name != 'run.log' for p in output.iterdir()):
        raise FileExistsError(f'refusing to overwrite {output}')
    parent = load_parent(source)
    config = replace(parent.config, episodes=len(parent.history)+additional_episodes,
                     eval_episodes=eval_episodes,
                     learning_rate=parent.config.learning_rate if learning_rate is None else learning_rate)
    record = run_record(parent, config, additional_episodes, train_env_seed, eval_seed)
    torch.set_num_threads(1)
    actor, optimizer, rng, env = restore_training(parent, config, train_env_seed)
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=True)
        (output/'config.json').write_text(json.dumps(record, indent=2)+'\n')
        (output/'training.csv').write_bytes(parent.history_bytes + (b'' if parent.history_bytes.endswith(b'\n') else b'\n'))
        new_rows, updates, train_seconds = train_more(parent, actor, optimizer, rng, env, config, output)
    finally:
        env.close()
    snapshot(output/'policy.pt', actor, optimizer, config, config.episodes, updates, rng)
    evaluation, frozen = evaluate_final(actor, config, output, eval_seed)
    all_rows = list(parent.history) + new_rows
    evaluation_stats = reward_statistics(evaluation) | {
        'parameters_unchanged': frozen, 'wins': sum(row['reward'] > 0 for row in evaluation),
        'win_rate': sum(row['reward'] > 0 for row in evaluation)/len(evaluation)}
    metrics = {'config': record, 'optimizer_updates': updates,
               'training': reward_statistics(all_rows) | {
                   'first100_mean': float(np.mean([row['reward'] for row in all_rows[:100]])),
                   'last100_mean': float(np.mean([row['reward'] for row in all_rows[-100:]]))},
               'continuation_training': reward_statistics(new_rows) | {'seconds': train_seconds},
               'evaluation': evaluation_stats, 'target_met': meets_target(evaluation_stats),
               'total_seconds': time.perf_counter()-started}
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    save_plots(output, 'pong', 'none', all_rows, evaluation)
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/pong_selected/policy.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--additional-episodes', type=int, default=1000)
    parser.add_argument('--train-env-seed', type=int, default=70001)
    parser.add_argument('--eval-seed', type=int, default=54321)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--eval-episodes', type=int, default=500)
    continue_training(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
