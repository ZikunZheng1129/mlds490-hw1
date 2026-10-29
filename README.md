# MLDS 490 Homework 1

Zikun Zheng

[Report](hw1_report.pdf)

## Files

| File | Contents |
|---|---|
| `shuttle.py` | Q2: shuttle MDP, finite-horizon solution, value iteration and policy iteration |
| `policy_gradient.py` | CartPole and the Part 2 EMA baseline experiments |
| `pong_tuning.py` | Initial training for the selected Pong configuration |
| `pong_continue.py` | Warm-start training from the saved Pong policy and Adam state |
| `results/` | Training records, evaluation scores, plots and saved models |
| `tests/` | Tests for the shuttle and policy gradient implementations |
| `build_report.py` | Generates the PDF from the saved results |

## Setup

Python 3.12. Run from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
# Q2: shuttle
python shuttle.py

# Policy Gradient Part 1
python policy_gradient.py --task cartpole --episodes 800 --output results/cartpole_plain
python pong_tuning.py --recipe ishan --episodes 3000 --seed 1 --output results/pong_initial_rerun
python pong_continue.py --source results/pong_initial_rerun/policy.pt --additional-episodes 1000 --train-env-seed 70001 --eval-seed 54321 --output results/pong_rerun

# Policy Gradient Part 2
python policy_gradient.py --task cartpole --baseline --episodes 800 --output results/cartpole_baseline
python policy_gradient.py --task pong --baseline --episodes 250 --learning-rate 0.001 --output results/pong_baseline
```

The report uses the saved Pong run in `results/pong_selected`: 256 hidden units,
4 games per update, 3000 initial training games followed by 1000 more games,
and 500 frozen evaluation games. The evaluation mean is 3.43, with 358/500 wins (71.6%).
Its update uses point-truncated discounted returns without the outer time discount.
The continuation restores the policy, Adam state and training RNG. It starts a fresh
environment with seed 70001 instead of restoring ALE state. Initial training uses
seed 1; the final evaluation uses seed 54321. Learning rate and batch size stay unchanged.
The two commands above repeat both stages; use fresh output paths for another run.
The configuration is based on [Ishan's implementation](https://github.com/ishan-monie/advanced-ml-hw1).

Alternatively, use the supplied 3000-game checkpoint to run only the continuation:

```bash
python pong_continue.py --source results/pong_initial/policy.pt --additional-episodes 1000 --train-env-seed 70001 --eval-seed 54321 --output results/pong_from_checkpoint
```

Part 2 keeps the EMA baseline runs required by the assignment. They use the original
32-unit networks, full-episode returns and seed 7, with 500 evaluations each.
They are separate baseline experiments, not a matched comparison with the selected Pong run.

## Validation and report

```bash
python policy_gradient.py --validate --output results/policy_validation.json
python -m pytest -q
python build_report.py
```

`build_report.py` reads the saved results and does not retrain the models.
