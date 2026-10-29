"""Make a plain homework report from the saved experiment results."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, PageBreak, Preformatted

ROOT = Path(__file__).resolve().parent
BODY = ParagraphStyle('body', fontName='Times-Roman', fontSize=11, leading=13.2, spaceAfter=7)
HEAD = ParagraphStyle('head', parent=BODY, fontName='Times-Bold', fontSize=12, spaceBefore=7)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=9.5, leading=11)
CODE = ParagraphStyle('code', fontName='Courier', fontSize=8, leading=10, spaceAfter=8)


def p(text, style=BODY):
    return Paragraph(text, style)


def load(path):
    return json.loads((ROOT / path).read_text())


def fig(path, width=252, height=137):
    return Image(str(ROOT / path), width=width, height=height, kind='proportional')


def pair(left, right):
    return Table([[fig(left), fig(right)]], colWidths=[252, 252],
                 style=[('LEFTPADDING', (0, 0), (-1, -1), 0),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 0)])


def table(rows):
    cells = [[p(escape(str(cell)), SMALL) for cell in row] for row in rows]
    item = Table(cells, colWidths=[504 / len(rows[0])] * len(rows[0]))
    item.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return item


def plain_plots():
    """Redraw the required plots in grayscale from the existing CSV data."""
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})
    shuttle = np.genfromtxt(ROOT/'results/shuttle/values_and_policy.csv', delimiter=',', names=True)
    for name, column, title in [('finite_horizon', 'J_finite_500', '2(a) Enumeration, T = 500'),
                                ('value_iteration', 'J_value_iteration', '2(b) Value iteration'),
                                ('optimal_policy', 'action', '2(c) Policy iteration')]:
        figure, ax = plt.subplots(figsize=(6.4, 3.5))
        ax.step(shuttle['queue'], shuttle[column], where='post', color='black')
        ax.set(xlabel='Customers waiting', ylabel='Action' if column == 'action' else 'Discounted cost J', title=title)
        if column == 'action':
            ax.set(yticks=[0, 1], yticklabels=['Wait', 'Dispatch'], ylim=(-0.1, 1.1))
            ax.axvline(13, color='gray', linestyle='--', linewidth=0.8)
        figure.tight_layout()
        figure.savefig(ROOT/f'results/shuttle/{name}.png', dpi=160)
        plt.close(figure)
    for task in ('cartpole', 'pong'):
        variants = ('plain', 'baseline', 'plain_1000') if task == 'pong' else ('plain', 'baseline')
        for variant in variants:
            folder = ROOT/f'results/{task}_{variant}'
            train = np.genfromtxt(folder/'training.csv', delimiter=',', names=True)
            scores = np.genfromtxt(folder/'evaluation.csv', delimiter=',', names=True)['reward']
            title = 'CartPole-v1' if task == 'cartpole' else 'Pong-v5'
            figure, ax = plt.subplots(figsize=(6.4, 3.5))
            ax.plot(train['episode'], train['reward'], color='0.7', linewidth=0.6, label='Episode reward')
            ax.plot(train['episode'][99:], np.convolve(train['reward'], np.ones(100)/100, 'valid'),
                    color='black', linewidth=1.2, label='100-episode average')
            ax.set(xlabel='Training episode', ylabel='Total reward', title=title)
            ax.legend(fontsize=9, frameon=False)
            figure.tight_layout()
            figure.savefig(folder/'training.png', dpi=160)
            plt.close(figure)
            figure, ax = plt.subplots(figsize=(6.4, 3.5))
            bins = np.arange(scores.min()-0.5, scores.max()+1.5) if task == 'pong' else 20
            ax.hist(scores, bins=bins, color='0.65', edgecolor='black', linewidth=0.5)
            ax.set(xlabel='Total reward', ylabel='Episodes', title=title+' (500 evaluations)')
            figure.tight_layout()
            figure.savefig(folder/'evaluation.png', dpi=160)
            plt.close(figure)


def shuttle_page():
    return [p('2.', HEAD),
        p('Let n be the number waiting (0 to 200), and let a = 0 mean wait and a = 1 mean dispatch. '
          'Dispatch happens before holding costs and new arrivals. One shuttle can be sent each period.'),
        Preformatted('q = max(n - 15a, 0);  c(n,a) = 100a + 2q\nn_next = min(200, q + A),  A uniform on {1,2,3,4,5}\nJ_t(n) = min_a [c(n,a) + 0.95 E J_(t+1)(n_next)]', CODE),
        p('J is discounted cost; reward value is V = -J. Excess arrivals are blocked without '
          'an extra penalty. For T=500, the terminal cost is zero.'),
        p('<b>2(a).</b> Enumerate all states, actions and arrivals by backward induction for 500 periods.'),
        p('<b>2(b).</b> Start value iteration at zero and stop when the largest change is below 1e-10.'),
        p('<b>2(c).</b> Alternate exact policy evaluation, (I - gamma P)J = c, and greedy improvement.'),
        pair('results/shuttle/finite_horizon.png', 'results/shuttle/value_iteration.png'),
        fig('results/shuttle/optimal_policy.png', width=300, height=158),
        p('<b>Result:</b> dispatch when n is at least 13; otherwise wait. '
          'J(0)=576.670, J(15)=676.670, and J(200)=3821.586. '
          'VI and PI give the same policy, with maximum cost difference 1.87e-9. '
          'The T=500 and infinite-horizon values differ by at most 4.94e-9.'),
        p('<b>Edge cases and checks:</b> max(n-15,0) handles partial loads. At n=200, waiting '
          'keeps the queue at 200; dispatching gives next states 186-190. Empty queues and ties choose wait. '
          'With gamma=0, dispatch costs 100 but saves at most 30 immediately, so all states wait '
          'and J(n)=2n exactly. Free dispatch serves every nonempty queue; free holding gives '
          'J=0 and waiting everywhere. Both match the computed results.'), PageBreak()]


def game_plots(variant, metrics):
    rows = [['Environment / training episodes', 'Eval mean', 'Eval SD']]
    plots = []
    for task in ('cartpole', 'pong'):
        key = f'{task}_{variant}'
        shown = 'pong_plain_1000' if task == 'pong' and variant == 'plain' else key
        stem = f'results/{shown}'
        label = '(a)' if task == 'cartpole' else '(b)'
        suffix = ' 1000-episode run.' if shown == 'pong_plain_1000' else ''
        plots += [p(('3' if variant == 'plain' else '') + label + '.' + suffix, SMALL),
                  pair(stem+'/training.png', stem+'/evaluation.png')]
        keys = [key, shown] if key != shown else [key]
        for item in keys:
            ev = metrics[item]['evaluation']
            name = 'CartPole-v1' if task == 'cartpole' else 'Pong-v5'
            rows.append([f'{name} / {metrics[item]["config"]["episodes"]}',
                         f'{ev["mean"]:.2f}', f'{ev["std_population"]:.2f}'])
    return plots + [table(rows)]


def plain_page(metrics):
    return [p('3. Part 1', HEAD),
        p('I use PyTorch REINFORCE with one Adam update after each episode. The objective is '
          'expected discounted episode reward. Actions are sampled from a two-output policy network.'),
        Preformatted('G_t = sum_(k=t)^(T-1) gamma^(k-t) r_k\nLoss = -sum_t gamma^t log pi(a_t|s_t) G_t', CODE),
        p('CartPole: 4 inputs, 32 tanh units, 2 outputs; gamma=0.95, learning rate=0.01, 800 episodes. '
          'Pong: 6400 inputs, 32 ReLU units, 2 outputs; gamma=0.99, learning rate=0.001 '
          '(a conservative stability choice). I first tried 250 episodes, then reran training '
          'for 1000 total episodes with the same seed and settings. The first 250 episodes '
          'matched the initial run.'),
        p('For ALE/Pong-v5, only actions 2 and 3 are used. I apply the supplied 80x80 binary '
          'preprocessing, then subtract the previous frame to provide motion information. '
          'The first difference is zero. Frame skip is 4 and sticky-action probability is 0.25.'),
        *game_plots('plain', metrics),
        p('Plots show total reward and a trailing 100-episode mean. Each model, including '
          'both Pong attempts, is evaluated for 500 full episodes with a frozen, stochastic '
          'policy. SD uses denominator 500. Training seed: 7; evaluation seeds: 20000-20499.', SMALL),
        p('<b>Checks and edge cases:</b> 50 random-policy episodes give means 20.16 for CartPole '
          'and -20.58 for Pong. The 1000-episode Pong run improves by only 0.12 points and still '
          'loses all 500 evaluations. Termination and time-limit truncation end episodes; terminal '
          'rewards are kept. Returns and frame history reset between episodes, not between Pong '
          'points. No return normalization or reward shaping is used.'), PageBreak()]


def baseline_page(metrics):
    cp = metrics['cartpole_baseline']['gradient_variance']
    po = metrics['pong_baseline']['gradient_variance']
    return [p('Policy Gradient: Part 2', HEAD),
        p('I repeat both experiments with an exponential moving average (EMA) baseline. '
          'It starts at zero and stays fixed for the current episode. After the policy update, '
          'b is replaced by 0.9b + 0.1 mean(G_t). Thus the baseline uses previous episodes '
          'and does not depend on the current action. The baseline comparison uses the original '
          'budgets: 800 CartPole and 250 Pong episodes, with the other settings unchanged.'),
        Preformatted('Loss = -sum_t gamma^t log pi(a_t|s_t) (G_t - b)', CODE),
        *game_plots('baseline', metrics),
        p('The baseline raises CartPole mean reward from 176.82 to 294.39 in this run. '
          'Pong stays near -20.5, so it is not solved. One seed and a short training budget '
          'are not enough to conclude that one method is always better.'),
        p('<b>Variance check:</b> for each final baseline-trained policy, I reuse its first '
          '20 evaluation episodes and compute gradients with b=0 and with the final EMA. '
          'The sample covariance trace falls from '
          f'{cp["plain"]["variance_trace"]:.2f} to {cp["baseline"]["variance_trace"]:.2f} '
          'for CartPole and from '
          f'{po["plain"]["variance_trace"]:.2f} to {po["baseline"]["variance_trace"]:.2f} '
          'for Pong. Lower gradient variance does not necessarily mean higher game scores.'),
        p('<b>Edge cases:</b> the first episode uses b=0. The EMA is updated only after '
          'the optimizer step, avoiding a baseline fitted to the current episode. '
          'It works with zero or negative rewards without dividing by a reward standard deviation. '
          'A constant baseline is unbiased, but a poor choice can increase variance.'), PageBreak()]


def final_page(args):
    github = ('Pending - add after upload.' if not args.github_url else
              f'{args.github_user} / {args.github_url.rstrip("/").split("/")[-1]} / {args.github_url}')
    return [p('<b>2. Validation:</b> I also checked a small deterministic problem: capacity 3, '
          'cap 8, arrivals 2, dispatch cost 5, holding cost 1, gamma=0.8 and horizon 4. '
          'Enumerating all 16 action sequences from each of 9 states agrees with dynamic '
          'programming to 3.55e-15. With no arrivals and n=0,...,15, the analytic cost '
          'min(40n,100) matches the algorithm to 1.89e-9.'),
        p('<b>Policy Gradient, Parts 1 and 2. Validation:</b> rewards [1,2,3] with gamma=0.5 give returns '
          '[2.75,3.5,3], matching hand calculation. In a one-step bandit with P(a=1)=0.3 '
          'and reward=a, exact enumeration gives gradient 0.21 with or without baseline 0.3. '
          'Its variance falls from 0.1029 to 0.0336 (67.3%). This tests the baseline idea '
          'independently of game performance. Evaluation leaves all model parameters unchanged.'),
        p('<b>Reproduce:</b> use Python 3.12 and run the commands below from the repository root. '
          'Create a virtual environment and install requirements.txt first.'),
        Preformatted('python3.12 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\npython shuttle.py\npython policy_gradient.py --task cartpole --episodes 800 \\\n  --learning-rate 0.01 --output results/cartpole_plain\npython policy_gradient.py --task cartpole --baseline --episodes 800 \\\n  --learning-rate 0.01 --output results/cartpole_baseline\npython policy_gradient.py --task pong --episodes 250 \\\n  --learning-rate 0.001 --output results/pong_plain\npython policy_gradient.py --task pong --episodes 1000 --learning-rate 0.001 \\\n  --output results/pong_plain_1000\npython policy_gradient.py --task pong --baseline --episodes 250 \\\n  --learning-rate 0.001 --output results/pong_baseline\npython policy_gradient.py --validate --output results/policy_validation.json\npython -m pytest -q\npython build_report.py', CODE),
        p('shuttle.py runs the parameter checks and exhaustive comparison. The --validate '
          'command runs the bandit, return and random-policy checks. Each training command '
          'also runs 500 evaluations and the 20-episode gradient check. Raw scores, metrics, '
          'first-10-step traces, models and plots are saved in results/.', SMALL),
        p('<b>With three more months:</b> I would train Pong longer, repeat both methods '
          'over several seeds, and compare a learned value baseline with the constant EMA. '
          'I would test learning rates and episode batch sizes, measure gradient variance '
          'with more samples, and check shuttle policies under different queue caps and '
          'arrival distributions.'),
        p('<b>GitHub username, repository and link:</b> '+escape(github), SMALL)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--github-user', default='ZikunZheng1129')
    parser.add_argument('--github-url', default='https://github.com/ZikunZheng1129/mlds490-hw1')
    args = parser.parse_args()
    names = ['cartpole_plain', 'cartpole_baseline', 'pong_plain', 'pong_baseline', 'pong_plain_1000']
    metrics = {name: load(f'results/{name}/metrics.json') for name in names}
    if any(m['evaluation']['episodes'] != 500 for m in metrics.values()):
        raise ValueError('Each experiment must contain 500 evaluation episodes.')
    plain_plots()
    story = shuttle_page() + plain_page(metrics) + baseline_page(metrics) + final_page(args)
    output = ROOT/'hw1_report.pdf'
    SimpleDocTemplate(str(output), pagesize=(612, 792), leftMargin=54, rightMargin=54,
                      topMargin=36, bottomMargin=36).build(story)
    print(output)


if __name__ == '__main__':
    main()
