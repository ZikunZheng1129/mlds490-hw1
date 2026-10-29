# MLDS 490 Homework 1

Zikun Zheng

[Report](hw1_report.pdf)

## Files

| File | Contents |
|---|---|
| `shuttle.py` | Q2: shuttle MDP, finite-horizon solution, value iteration and policy iteration |
| `policy_gradient.py` | Policy Gradient Parts 1 and 2: CartPole and Pong, with and without an EMA baseline |
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
python policy_gradient.py --task pong --episodes 250 --learning-rate 0.001 --output results/pong_plain
python policy_gradient.py --task pong --episodes 1000 --learning-rate 0.001 --output results/pong_plain_1000

# Policy Gradient Part 2
python policy_gradient.py --task cartpole --baseline --episodes 800 --output results/cartpole_baseline
python policy_gradient.py --task pong --baseline --episodes 250 --learning-rate 0.001 --output results/pong_baseline
```

Each run uses seed 7 and evaluates the trained policy for 500 episodes. For Pong,
I first tried 250 training episodes, then repeated the plain policy gradient run
with 1000 total episodes. The Pong baseline comparison uses 250 episodes for both methods.

## Validation and report

```bash
python policy_gradient.py --validate --output results/policy_validation.json
python -m pytest -q
python build_report.py
```

`build_report.py` reads the saved results and does not retrain the models.
