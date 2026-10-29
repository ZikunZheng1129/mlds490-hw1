"""Checks of the local Pong comparison runner, independent of long training runs."""
import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

import pong_tuning as tuning
from policy_gradient import policy_loss


def test_point_returns_and_uniform_time_weighting():
    rewards = [0, 1, 0, -1]
    np.testing.assert_allclose(tuning.returns_to_go(rewards, .5, False), [.375, .75, -.5, -1])
    np.testing.assert_allclose(tuning.returns_to_go(rewards, .5, True), [.5, 1, -.5, -1])
    logp = torch.tensor([-.2, -.3], requires_grad=True)
    returns = torch.tensor([2., 3.])
    practical = -(logp * returns).sum()
    strict = policy_loss(logp, returns, .5)
    assert torch.autograd.grad(practical, logp, retain_graph=True)[0].tolist() == [-2., -3.]
    assert torch.autograd.grad(strict, logp)[0].tolist() == [-2., -1.5]


def test_batch_divisors_preserve_expected_gradient_scale():
    states = np.zeros((2, 6400), dtype=np.float32)
    states[:, 0] = 1
    games = [{'states': states, 'actions': [0, 1], 'rewards': [0., 1.]}] * 2
    gradients = []
    for divisor in ('sum', 'episodes', 'steps'):
        torch.manual_seed(5)
        actor = tuning.make_actor(tuning.recipe_config('uniform-only'))
        optimizer = torch.optim.SGD(actor.parameters(), lr=0)
        config = replace(tuning.recipe_config('uniform-only'), loss_divisor=divisor, chunk_size=1)
        tuning.update(actor, optimizer, games, config)
        gradients.append(torch.cat([p.grad.ravel() for p in actor.parameters()]))
    torch.testing.assert_close(gradients[0], gradients[1] * 2)
    torch.testing.assert_close(gradients[0], gradients[2] * 4)


def test_uniform_runner_gives_late_action_undiscounted_time_weight():
    actor = torch.nn.Linear(6400, 2)
    torch.nn.init.zeros_(actor.weight)
    torch.nn.init.zeros_(actor.bias)
    game = {'states': np.zeros((2, 6400), dtype=np.int8),
            'actions': [0, 1], 'rewards': [0., 1.]}
    tuning.update(actor, torch.optim.SGD(actor.parameters(), lr=0),
                  [game], tuning.recipe_config('uniform-only'))
    # At p=1/2, -[.99 log p(a0) + 1 log p(a1)] has this exact derivative.
    # Multiplying the second term by the original outer .99 would cancel it.
    torch.testing.assert_close(actor.bias.grad, torch.tensor([.005, -.005]))


class TinyPong:
    """Two-decision episodic simulator with observable reset and action behavior."""
    def __init__(self):
        self.unwrapped = self
        self.actions = []
        self.seeds = []
        self.action_space = type('Space', (), {'seed': lambda self, seed: None})()

    def get_action_meanings(self):
        return ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.steps = 0
        return self.frame(), {}

    def frame(self):
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame[41, 20, 0] = 255
        return frame

    def step(self, action):
        self.actions.append(action)
        if action == 1:
            return self.frame(), 0., False, False, {}
        self.steps += 1
        return self.frame(), float(self.steps == 2), self.steps == 2, False, {}

    def close(self):
        pass


@pytest.mark.parametrize('recipe', ['uniform-only', 'yiping', 'ishan'])
def test_protocol_partial_batch_checkpoint_and_frozen_evaluation(tmp_path, monkeypatch, recipe):
    environments = []

    def make_env(task):
        env = TinyPong()
        environments.append(env)
        return env

    monkeypatch.setattr(tuning, 'make_env', make_env)
    config = replace(tuning.recipe_config(recipe), episodes=3, eval_episodes=2)
    metrics = tuning.train(config, tmp_path)
    training = list(csv.DictReader((tmp_path/'training.csv').open()))
    evaluation = list(csv.DictReader((tmp_path/'evaluation.csv').open()))
    assert len(training) == 3 and len(evaluation) == 2
    assert metrics['optimizer_updates'] == (3 if recipe == 'uniform-only' else 1)
    assert metrics['evaluation']['parameters_unchanged']
    assert metrics['evaluation']['mean'] == 1
    saved = torch.load(tmp_path/'policy.pt', weights_only=True)
    assert saved['completed_episodes'] == 3
    assert 'optimizer_state_dict' in saved and 'torch_rng_state' in saved
    assert (tmp_path/'checkpoint_00003.pt').is_file()
    assert json.loads((tmp_path/'config.json').read_text())['outer_time_discount'] is False
    all_actions = set(environments[0].actions)
    assert all_actions <= ({1, 2, 3} if recipe == 'yiping' else {2, 3})
    assert metrics['training_startup_fire_actions'] == (3 if recipe == 'yiping' else 0)
    if recipe == 'ishan':
        assert environments[0].seeds == [1, None, None, None]
    with pytest.raises(FileExistsError):
        tuning.train(config, tmp_path)


def test_recipe_validation_and_initial_frame_protocol():
    with pytest.raises(ValueError):
        tuning.recipe_config('missing')
    for changes in ({'episodes': 0}, {'eval_episodes': 0}, {'batch_episodes': 0}, {'learning_rate': 0.}):
        with pytest.raises(ValueError):
            tuning.validate_config(replace(tuning.recipe_config('uniform-only'), **changes))
    for recipe in ('yiping', 'ishan'):
        config = tuning.recipe_config(recipe)
        actor = tuning.make_actor(config)
        trajectory = tuning.collect(actor, TinyPong(), config, 1, np.random.default_rng(7))
        assert np.count_nonzero(trajectory['states'][0]) == (1 if recipe == 'yiping' else 0)
