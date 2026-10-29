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
        variants = ('selected', 'baseline') if task == 'pong' else ('plain', 'baseline')
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
        key = 'pong_selected' if task == 'pong' and variant == 'plain' else f'{task}_{variant}'
        stem = f'results/{key}'
        label = '(a)' if task == 'cartpole' else '(b)'
        plots += [p(('3' if variant == 'plain' else '') + label + '.', SMALL),
                  pair(stem+'/training.png', stem+'/evaluation.png')]
        ev = metrics[key]['evaluation']
        name = 'CartPole-v1' if task == 'cartpole' else 'Pong-v5'
        rows.append([f'{name} / {metrics[key]["config"]["episodes"]}',
                     f'{ev["mean"]:.2f}', f'{ev["std_population"]:.2f}'])
    return plots + [table(rows)]


def plain_page(metrics):
    pong = metrics['pong_selected']
    config, evaluation = pong['config'], pong['evaluation']
    warm = config['warm_start']
    train_count, eval_count = pong['training']['episodes'], evaluation['episodes']
    wins = evaluation['wins']
    return [p('3. Part 1', HEAD),
        p('<b>3(a).</b> CartPole uses a 4-input network with 32 tanh units and 2 outputs. '
          'I train for 800 episodes with Adam, learning rate 0.01 and gamma=0.95, '
          'using one update per episode and discounted reward-to-go.'),
        Preformatted('G_t = sum_(k=t)^(T-1) gamma^(k-t) r_k\nCartPole loss = -sum_t gamma^t log pi(a_t|s_t) G_t', CODE),
        p('<b>3(b).</b> Pong uses 6400 inputs, 256 ReLU units and one Bernoulli logit for '
          f'actions 2 and 3. I train for {train_count} episodes '
          f'({warm["inherited_episodes"]} initial + {warm["additional_episodes"]} continued), '
          'updating after every 4 games with Adam, learning rate 0.001 and gamma=0.99. '
          'The smaller learning rate is used for stability. Warm-start training restores '
          'the policy, Adam state and training RNG, but restarts the environment without '
          'restoring ALE state. Frames use the supplied 80x80 binary preprocessing and consecutive-frame '
          'differences; the first difference is zero. Frame skip is 4 and sticky-action probability is 0.25.'),
        p('For Pong, I truncate reward-to-go at the next scoring point and omit the outer '
          'time discount in the policy loss. This is a practical approximation to the full '
          'discounted episodic objective. Entire games are still used for training scores and evaluation.'),
        Preformatted('Pong loss = -(1/4) sum_games sum_t log pi(a_t|s_t) G_t(point)', CODE),
        *game_plots('plain', metrics),
        p('Both training plots include the trailing 100-episode mean. Each final stochastic policy is '
          f'evaluated for {eval_count} full games without updates; SD uses denominator {eval_count}. '
          f'Pong averages {evaluation["mean"]:.2f} points and wins {wins}/{eval_count} games '
          f'({100*wins/eval_count:.1f}%). CartPole uses training seed 7 and evaluation seeds '
          f'20000-20499. Pong uses initial training seed {config["seed"]}, continuation environment '
          f'seed {warm["train_env_seed"]} and evaluation seed {warm["evaluation_seed"]}.', SMALL),
        p('<b>Checks and edge cases:</b> termination or truncation ends a rollout, and the final '
          'reward is retained. Frame history resets each game. Pong returns reset at each score; '
          f'zero-reward steps inherit the discounted next score. All {train_count} Pong training games and '
          f'{eval_count} evaluations terminated normally, with no truncation and no evaluation parameter changes.', SMALL),
        PageBreak()]


def baseline_page(metrics):
    cp = metrics['cartpole_baseline']['gradient_variance']
    po = metrics['pong_baseline']['gradient_variance']
    return [p('Policy Gradient: Part 2', HEAD),
        p('I use an exponential moving average (EMA) baseline for both environments. It starts '
          'at zero and stays fixed during an episode. After the policy update, '
          'b is replaced by 0.9b + 0.1 mean(G_t), using previous episodes without depending '
          'on the current action.'),
        p('These baseline runs use 32 hidden units (tanh for CartPole, ReLU for Pong), '
          'two policy outputs, full-episode reward-to-go and one update per episode. '
          'CartPole uses 800 episodes, gamma=0.95 and learning rate 0.01; Pong uses 250 episodes, '
          'gamma=0.99 and learning rate 0.001. Training seed is 7 and evaluation seeds are '
          '20000-20499. These retain the original baseline setup; they are not a matched '
          'comparison with the Part 1 Pong configuration.'),
        Preformatted('Loss = -sum_t gamma^t log pi(a_t|s_t) (G_t - b)', CODE),
        *game_plots('baseline', metrics),
        p('Each policy is frozen for 500 evaluation episodes. The CartPole baseline policy '
          'balances for 294.39 steps on average. The Pong baseline policy averages -20.53 '
          'points and is not competitive.'),
        p('<b>Variance check:</b> for each final baseline policy, I use the same 20 evaluation '
          'trajectories to compute gradients with b=0 and with the final EMA. The empirical '
          'covariance trace changes from '
          f'{cp["plain"]["variance_trace"]:.2f} to {cp["baseline"]["variance_trace"]:.2f} '
          'for CartPole and from '
          f'{po["plain"]["variance_trace"]:.2f} to {po["baseline"]["variance_trace"]:.2f} '
          'for Pong. This directly checks variance reduction on fixed trajectories.'),
        p('<b>Edge cases:</b> the first episode uses b=0. Updating the EMA only after '
          'the optimizer step avoids fitting the baseline to the current episode. '
          'Zero and negative rewards do not require division by a reward standard deviation. '
          'A constant baseline is unbiased, but a poor choice can increase variance.'), PageBreak()]


def final_page(args):
    github = ('Pending - add after upload.' if not args.github_url else
              f'{args.github_user} / {args.github_url.rstrip("/").split("/")[-1]} / {args.github_url}')
    commands = [
        'python3.12 -m venv .venv',
        'source .venv/bin/activate',
        'python -m pip install -r requirements.txt',
        'python shuttle.py',
        'python policy_gradient.py --task cartpole --episodes 800',
        '  --learning-rate 0.01 --output results/cartpole_plain',
        'python policy_gradient.py --task cartpole --baseline --episodes 800',
        '  --learning-rate 0.01 --output results/cartpole_baseline',
        'python pong_tuning.py --recipe ishan --episodes 3000 --seed 1',
        '  --output results/pong_initial_rerun',
        'python pong_continue.py --source results/pong_initial_rerun/policy.pt',
        '  --additional-episodes 1000 --train-env-seed 70001 --eval-seed 54321',
        '  --output results/pong_rerun',
        'python policy_gradient.py --task pong --baseline --episodes 250',
        '  --learning-rate 0.001 --output results/pong_baseline',
        'python policy_gradient.py --validate --output results/policy_validation.json',
        'python -m pytest -q',
        'python build_report.py',
    ]
    commands = [line + (' ' + chr(92) if i+1 < len(commands) and commands[i+1].startswith('  ') else '')
                for i, line in enumerate(commands)]
    return [p('<b>2. Validation:</b> I checked a deterministic problem with capacity 3, cap 8, '
          'arrivals 2, dispatch cost 5, holding cost 1, gamma=0.8 and horizon 4. Enumerating '
          'all 16 action sequences from each of 9 states agrees with dynamic programming '
          'to 3.55e-15. With no arrivals and n=0,...,15, the analytic cost min(40n,100) '
          'matches the algorithm to 1.89e-9.'),
        p('<b>Policy Gradient, Parts 1 and 2. Validation:</b> rewards [1,2,3] with gamma=0.5 '
          'give full returns [2.75,3.5,3]. For Pong, rewards [0,1,0,-1] with gamma=0.5 give '
          'point-truncated returns [0.5,1,-0.5,-1]. Both match hand calculation. In a one-step '
          'bandit with P(a=1)=0.3 and reward=a, exact enumeration gives gradient 0.21 with '
          'or without baseline 0.3. Its variance falls from 0.1029 to 0.0336 (67.3%). '
          'This tests the estimator independently of game performance. All reported evaluations '
          'leave model parameters unchanged.'),
        p('<b>Reproduce:</b> use Python 3.12 and run these commands from the repository root. '
          'The selected Pong model and scores are in results/pong_selected. The two Pong commands '
          'reproduce the initial training and warm-start continuation; use fresh output paths. '
          'The README also gives a shortcut using the supplied initial checkpoint. '
          'build_report.py uses saved results and does not retrain.'),
        Preformatted('\n'.join(commands), CODE),
        p('shuttle.py runs the deterministic and exhaustive checks. The --validate command '
          'checks full returns, the bandit and random policies. The tests also verify '
          'point-truncated returns and the four-game update. Training commands save raw scores, '
          'metrics, models and plots, followed by 500 frozen evaluations. The baseline runner '
          'additionally saves the 20-trajectory variance check and first-10-step trace.', SMALL),
        p('<b>With three more months:</b> I would repeat training over multiple seeds, test '
          'longer Pong training, use identical configurations for baseline ablations, '
          'measure gradient variance over more trajectories, and test shuttle policies '
          'under different queue caps and arrival distributions.'),
        p('<b>Pong configuration reference:</b> '
          '<link href="https://github.com/ishan-monie/advanced-ml-hw1">'
          'github.com/ishan-monie/advanced-ml-hw1</link>. The implementation and reported runs are local.', SMALL),
        p('<b>GitHub username, repository and link:</b> '+escape(github), SMALL)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--github-user', default='ZikunZheng1129')
    parser.add_argument('--github-url', default='https://github.com/ZikunZheng1129/mlds490-hw1')
    args = parser.parse_args()
    names = ['cartpole_plain', 'cartpole_baseline', 'pong_selected', 'pong_baseline']
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
