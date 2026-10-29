"""Controlled checks of the estimator, preprocessing, and frozen evaluation."""
import numpy as np
import pytest
import torch

from policy_gradient import (
    Actor, EmaBaseline, discounted_returns, evaluate, policy_loss,
    preprocess, toy_bandit_validation,
)


def test_discounted_returns_hand_calculation_and_gamma_zero():
    np.testing.assert_allclose(discounted_returns([1, 2, 3], 0.5), [2.75, 3.5, 3])
    np.testing.assert_allclose(discounted_returns([-1, 0, 1], 0), [-1, 0, 1])
    np.testing.assert_allclose(discounted_returns([0, 1, 0, -1], 1), [0, 0, -1, -1])
    assert discounted_returns([], 0.95).size == 0
    with pytest.raises(ValueError):
        discounted_returns([1], 1.1)


def test_exact_discounted_episode_loss():
    log_prob = torch.tensor([-0.2, -0.3, -0.4])
    returns = torch.tensor([2.75, 3.5, 3.0])
    expected = -(log_prob * torch.tensor([1.0, 0.5, 0.25]) * (returns - 1.0)).sum()
    assert torch.allclose(policy_loss(log_prob, returns, 0.5, 1.0), expected)


def test_baseline_is_past_only_and_immutable():
    before = EmaBaseline(value=2.0, decay=0.9)
    after = before.updated(np.array([4.0, 6.0]))
    assert before.value == 2.0
    assert after.value == pytest.approx(2.3)


def test_baseline_preserves_bandit_gradient_and_reduces_variance():
    result = toy_bandit_validation()
    assert result['plain_expected_gradient'] == pytest.approx(0.21)
    assert result['baseline_expected_gradient'] == pytest.approx(0.21)
    assert result['baseline_gradient_variance'] < result['plain_gradient_variance']


def test_positive_reward_moves_probability_toward_selected_action():
    torch.manual_seed(8)
    actor = Actor(4, 8)
    states = torch.tensor([[0.1, -0.2, 0.3, 0.0]])
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    before = actor(states).softmax(-1)[0, 1].item()
    loss = policy_loss(actor(states).log_softmax(-1)[:, 1], torch.tensor([2.0]), 0.95)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert actor(states).softmax(-1)[0, 1].item() > before


def test_preprocess_removes_background_without_modifying_frame():
    frame = np.full((210, 160, 3), 144, dtype=np.uint8)
    frame[40, 20, 0] = 255
    frame[41, 21, 0] = 109
    original = frame.copy()
    pixels = preprocess(frame)
    assert pixels.shape == (80, 80)
    assert set(np.unique(pixels)) <= {0, 1}
    assert pixels.dtype == np.float32
    np.testing.assert_array_equal(frame, original)


def test_evaluation_is_frozen_and_seed_reproducible():
    torch.manual_seed(13)
    actor = Actor(4, 8)
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    a = evaluate(actor, 'cartpole', episodes=3, seed=91)
    b = evaluate(actor, 'cartpole', episodes=3, seed=91)
    assert a == b
    for key, value in actor.state_dict().items():
        assert torch.equal(value, before[key])
    assert all(1 <= row['length'] <= 500 for row in a)


@pytest.mark.parametrize('use_baseline', [False, True])
def test_end_to_end_saved_model_reproduces_evaluation(tmp_path, use_baseline):
    """Tiny genuine CartPole train/eval run, including checkpoint and report data."""
    from argparse import Namespace
    import csv
    import json
    from policy_gradient import train

    args = Namespace(task='cartpole', baseline=use_baseline, episodes=4,
                     eval_episodes=20, seed=17, eval_seed=1700, hidden=8,
                     learning_rate=0.01, output=str(tmp_path), validate=False)
    train(args)
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['evaluation']['parameters_unchanged']
    assert metrics['gradient_variance']['episodes'] == 20
    checkpoint = torch.load(tmp_path/'policy.pt', weights_only=True)
    restored = Actor(4, 8)
    restored.load_state_dict(checkpoint['state_dict'])
    repeated = evaluate(restored, 'cartpole', 20, 1700)
    with (tmp_path/'evaluation.csv').open() as file:
        saved = list(csv.DictReader(file))
    np.testing.assert_equal([float(row['reward']) for row in saved],
                            [row['reward'] for row in repeated])
    assert metrics['evaluation']['mean'] == pytest.approx(np.mean([row['reward'] for row in repeated]))
    assert (tmp_path/'training.png').stat().st_size > 1000
    assert (tmp_path/'evaluation.png').stat().st_size > 1000


def test_pong_full_episode_uses_only_allowed_actions():
    from policy_gradient import episode, make_env
    torch.set_num_threads(1)
    env = make_env('pong')
    try:
        assert env.spec.id == 'ALE/Pong-v5'
        trajectory = episode(Actor(6400, 8), env, 'pong', 23, np.random.default_rng(23))
    finally:
        env.close()
    assert set(trajectory['actions']) <= {0, 1}  # mapped to ALE 2 and 3
    assert {row['action'] for row in trajectory['trace']} <= {2, 3}
    assert set(trajectory['rewards']) <= {-1, 0, 1}
    assert -21 <= sum(trajectory['rewards']) <= 21
    assert trajectory['terminated'] or trajectory['truncated']


def test_value_baseline_is_detached_from_policy_gradient():
    from policy_gradient import ValueNetwork

    torch.manual_seed(10)
    actor, critic = Actor(4, 8), ValueNetwork(4, 8)
    states = torch.tensor([[0.1, 0.0, -0.2, 0.3], [0.0, 0.5, 0.1, -0.1]])
    values = critic(states)
    loss = policy_loss(actor(states).log_softmax(-1)[:, 0],
                       torch.tensor([2.0, 1.0]), 0.95, values)
    loss.backward()
    assert any(parameter.grad is not None for parameter in actor.parameters())
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_value_network_fits_monte_carlo_targets_and_preserves_rng():
    from policy_gradient import make_value_network, fit_value_network

    torch.manual_seed(20)
    rng_before = torch.random.get_rng_state().clone()
    critic = make_value_network(4, 8, seed=20)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    states = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]])
    targets = torch.tensor([-2., -1., 1., 2.])
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.03)
    before = float(torch.nn.functional.mse_loss(critic(states), targets).detach())
    for _ in range(50):
        fit_value_network(critic, optimizer, states, targets)
    after = float(torch.nn.functional.mse_loss(critic(states), targets).detach())
    assert after < before * 0.05


def test_state_dependent_baseline_preserves_exact_expected_gradient():
    """Enumerate two states and both actions, including their sampling weights."""
    logits = torch.tensor([0.3, -0.4], requires_grad=True)
    probabilities = logits.sigmoid()
    action_probabilities = torch.stack((1-probabilities, probabilities), dim=1)
    weights = (torch.tensor([0.4, 0.6])[:, None] * action_probabilities).detach().ravel()
    log_probabilities = action_probabilities.log().ravel()
    rewards = torch.tensor([0., 1., 2., -1.])
    state_values = torch.tensor([0.3, 1.0]).repeat_interleave(2)
    plain = policy_loss(weights * log_probabilities, rewards, 1.)
    baseline = policy_loss(weights * log_probabilities, rewards, 1., state_values)
    grad_plain = torch.autograd.grad(plain, logits, retain_graph=True)[0]
    grad_baseline = torch.autograd.grad(baseline, logits)[0]
    expected = -torch.tensor([0.4, -1.8]) * probabilities * (1-probabilities)
    torch.testing.assert_close(grad_plain, expected)
    torch.testing.assert_close(grad_baseline, expected)


def test_value_baseline_end_to_end_checkpoint_and_frozen_diagnostics(tmp_path):
    from argparse import Namespace
    import csv
    import json
    from policy_gradient import ValueNetwork, train

    args = Namespace(task='cartpole', baseline=False, value_baseline=True,
                     value_learning_rate=0.001, episodes=4, eval_episodes=20,
                     seed=17, eval_seed=1700, hidden=8, learning_rate=0.01,
                     output=str(tmp_path), validate=False)
    train(args)
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['config']['baseline_type'] == 'value'
    assert metrics['evaluation']['parameters_unchanged']
    assert metrics['evaluation']['value_parameters_unchanged']
    assert metrics['gradient_variance']['baseline_type'] == 'value'
    actor = Actor(4, 8)
    actor.load_state_dict(torch.load(tmp_path/'policy.pt', weights_only=True)['state_dict'])
    critic = ValueNetwork(4, 8)
    critic.load_state_dict(torch.load(tmp_path/'value.pt', weights_only=True)['state_dict'])
    trace = json.loads((tmp_path/'first10_steps.json').read_text())
    states = torch.tensor([row['state'] for row in trace], dtype=torch.float32)
    np.testing.assert_allclose([row['baseline_candidate'] for row in trace],
                               critic(states).detach().numpy(), atol=1e-7)
    repeated = evaluate(actor, 'cartpole', 20, 1700)
    with (tmp_path/'evaluation.csv').open() as file:
        saved = list(csv.DictReader(file))
    assert [float(row['reward']) for row in saved] == [row['reward'] for row in repeated]
    with (tmp_path/'training.csv').open() as file:
        history = list(csv.DictReader(file))
    assert len(history) == 4
    assert all(np.isfinite(float(row['value_loss'])) for row in history)
