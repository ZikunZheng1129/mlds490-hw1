# MLDS 490 Homework 1

GitHub: [ZikunZheng1129/mlds490-hw1](https://github.com/ZikunZheng1129/mlds490-hw1).

Read the [PDF report](hw1_report.pdf). Question 2 is in `shuttle.py`;
Policy Gradient Parts 1 and 2 are in `policy_gradient.py`;
saved data, models, and plots are in `results/`.

Use Python 3.12 and run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python shuttle.py
python policy_gradient.py --task cartpole --episodes 800 --output results/cartpole_plain
python policy_gradient.py --task cartpole --baseline --episodes 800 --output results/cartpole_baseline
python policy_gradient.py --task pong --episodes 250 --learning-rate 0.001 --output results/pong_plain
python policy_gradient.py --task pong --episodes 1000 --learning-rate 0.001 --output results/pong_plain_1000
python policy_gradient.py --task pong --baseline --episodes 250 --learning-rate 0.001 --output results/pong_baseline
python policy_gradient.py --validate --output results/policy_validation.json
python -m pytest -q
python build_report.py
```

All experiments have already been run;
`python build_report.py` alone rebuilds the report. Each training run uses seed 7
and evaluates 500 episodes on new seeds. Pong without a baseline was first tried
for 250 episodes, then rerun for 1000 total episodes using the same settings.
Both attempts remain weak; the baseline comparison uses the original 250-episode budget.
The report explains the validation experiments and their results.

The report includes the GitHub username and repository link. Submit `hw1_report.pdf`
on Canvas.
