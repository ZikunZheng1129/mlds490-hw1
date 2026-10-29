"""Warm-start continuation and honest frozen evaluation checks."""
import csv
from dataclasses import replace
import hashlib
import json
import sys

import numpy as np
import pytest
import torch

import pong_continue as continuation
import pong_tuning as tuning


class TinyPong:
    def __init__(self):
        self.seeds = []
        self.actions = []
        self.closed = False

    def reset(self, seed=None):
        self.seeds.append(seed)
        torch.rand(1)  # Initialization must not replace the saved action RNG.
        self.steps = 0
        return self.frame(), {}

    def frame(self):
        image = np.zeros((210, 160, 3), dtype=np.uint8)
        image[41, 20 + 2 * self.steps, 0] = 255
        return image

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        return self.frame(), float(self.steps == 2), self.steps == 2, False, {}

    def close(self):
        self.closed = True


@pytest.fixture
def parent_run(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(tuning, 'make_env', lambda task: TinyPong())
    monkeypatch.setattr(tuning, 'save_plots', lambda *args: None)
    path = tmp_path/'parent'
    tuning.train(replace(tuning.recipe_config('ishan'), episodes=4,
                         hidden=8, eval_episodes=2), path)
    return path


def test_restores_actor_adam_and_rng_after_environment_initialization(parent_run, monkeypatch):
    monkeypatch.setattr(continuation, 'make_env', lambda task: TinyPong())
    parent = continuation.load_parent(parent_run/'policy.pt')
    config = replace(parent.config, episodes=8)
    actor, optimizer, rng, env = continuation.restore_training(parent, config, 70001)
    checkpoint = parent.checkpoint
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, checkpoint['state_dict'][key], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), checkpoint['torch_rng_state'])
    assert rng.bit_generator.state == checkpoint['numpy_rng_state']
    for key, value in optimizer.state_dict()['state'].items():
        for field, tensor in value.items():
            torch.testing.assert_close(tensor, checkpoint['optimizer_state_dict']['state'][key][field])
    assert env.seeds == [70001]
    env.close()
    lower_lr = replace(config, learning_rate=.0005)
    _, changed_optimizer, _, env = continuation.restore_training(parent, lower_lr, 70002)
    assert changed_optimizer.param_groups[0]['lr'] == .0005
    assert int(next(iter(changed_optimizer.state.values()))['step']) == 1
    env.close()


def test_continuation_appends_history_streams_evaluation_and_preserves_source(parent_run, tmp_path, monkeypatch):
    environments = []
    def make_env(task):
        env = TinyPong()
        environments.append(env)
        return env
    monkeypatch.setattr(continuation, 'make_env', make_env)
    monkeypatch.setattr(continuation, 'save_plots', lambda *args: None)
    monkeypatch.setattr(continuation, 'CHECKPOINT_EVERY', 4)
    monkeypatch.setattr(continuation, 'LOG_EVERY', 2)
    output = tmp_path/'continued'
    source_bytes = (parent_run/'training.csv').read_bytes()
    source_hash = hashlib.sha256((parent_run/'policy.pt').read_bytes()).hexdigest()
    original_collect = continuation.collect
    seen = []
    def collect(*args, **kwargs):
        if kwargs.get('evaluating', False):
            number = args[3]
            assert len((output/'evaluation.csv').read_text().splitlines()) == number
            seen.append(number)
        return original_collect(*args, **kwargs)
    monkeypatch.setattr(continuation, 'collect', collect)
    metrics = continuation.continue_training(parent_run/'policy.pt', output,
                                             additional_episodes=5, eval_episodes=2)
    rows = list(csv.DictReader((output/'training.csv').open()))
    assert (output/'training.csv').read_bytes().startswith(source_bytes)
    assert [int(r['episode']) for r in rows] == list(range(1, 10))
    assert [int(r['update']) for r in rows] == [1]*4 + [2]*4 + [3]
    assert len(list(csv.DictReader((output/'evaluation.csv').open()))) == 2
    assert seen == [1, 2]
    assert environments[0].seeds == [70001] + [None]*5
    assert environments[1].seeds == [54321] + [None]*2
    assert all(env.closed and set(env.actions) <= {2, 3} for env in environments)
    assert metrics['training']['episodes'] == 9
    assert metrics['continuation_training']['episodes'] == 5
    assert metrics['optimizer_updates'] == 3
    assert metrics['evaluation']['parameters_unchanged']
    assert metrics['evaluation']['mean'] == 1 and metrics['evaluation']['wins'] == 2
    assert not metrics['target_met']  # A smoke evaluation is not a 500-game result.
    saved = torch.load(output/'policy.pt', weights_only=True)
    last = torch.load(output/'checkpoint_00009.pt', weights_only=True)
    assert saved['completed_episodes'] == 9 and saved['optimizer_updates'] == 3
    assert (output/'checkpoint_00008.pt').exists()
    assert not saved['exact_resume_supported']
    assert torch.equal(saved['torch_rng_state'], last['torch_rng_state'])
    assert not torch.equal(saved['torch_rng_state'], torch.get_rng_state())
    config = json.loads((output/'config.json').read_text())
    assert config['warm_start']['source_sha256'] == source_hash
    assert config['warm_start']['inherited_episodes'] == 4
    assert config['warm_start']['additional_episodes'] == 5
    assert not config['warm_start']['exact_resume']
    assert (parent_run/'training.csv').read_bytes() == source_bytes
    assert hashlib.sha256((parent_run/'policy.pt').read_bytes()).hexdigest() == source_hash
    with pytest.raises(FileExistsError):
        continuation.continue_training(parent_run/'policy.pt', output, additional_episodes=1)


def test_parent_history_must_match_checkpoint(parent_run):
    path = parent_run/'training.csv'
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines[:-1])+'\n')
    with pytest.raises(ValueError, match='history'):
        continuation.load_parent(parent_run/'policy.pt')


@pytest.mark.parametrize('changes', [
    {'additional_episodes':0}, {'eval_episodes':0}, {'train_env_seed':-1},
    {'eval_seed':-1}, {'learning_rate':0.}, {'learning_rate':float('nan')},
])
def test_invalid_options_fail_before_output(parent_run, tmp_path, changes):
    output = tmp_path/'invalid'
    with pytest.raises(ValueError):
        continuation.continue_training(parent_run/'policy.pt', output, **changes)
    assert not output.exists()


def test_target_requires_both_strict_thresholds_and_500_complete_frozen_games():
    result = {'episodes':500,'mean':.01,'wins':251,'parameters_unchanged':True,
              'terminated':500,'truncated':0}
    assert continuation.meets_target(result)
    for changes in [{'mean':0.}, {'wins':250}, {'episodes':499},
                    {'parameters_unchanged':False}, {'truncated':1}, {'terminated':499}]:
        assert not continuation.meets_target(result | changes)


def test_cli_defaults_and_overrides(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(continuation, 'continue_training', lambda **kw: calls.append(kw))
    monkeypatch.setattr(sys, 'argv', ['pong_continue.py','--output',str(tmp_path)])
    continuation.main()
    assert str(calls[-1]['source']) == 'results/pong_selected/policy.pt'
    assert calls[-1]['additional_episodes'] == 1000 and calls[-1]['eval_episodes'] == 500
    monkeypatch.setattr(sys, 'argv', ['pong_continue.py','--output',str(tmp_path),
        '--additional-episodes','12','--train-env-seed','88','--eval-seed','99','--learning-rate','.0005'])
    continuation.main()
    assert calls[-1]['additional_episodes'] == 12
    assert calls[-1]['learning_rate'] == .0005 and calls[-1]['eval_seed'] == 99
