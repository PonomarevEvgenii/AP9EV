from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent / 'U1_results'
OUTPUT.mkdir(exist_ok=True)
RUNS = 10
DIMS = [10, 30, 100]
PROBLEMS = ['OneMax', 'LeadingOnes']
BASE = dict(pop_size=50, elite=0.15, crossover_prob=0.85, selection='rank', crossover='one_point', mutation=0.01)
TESTS = [
    ('zakladni', {}),
    ('pop_30', {'pop_size': 30}),
    ('pop_100', {'pop_size': 100}),
    ('elite_10', {'elite': 0.10}),
    ('elite_20', {'elite': 0.20}),
    ('crossprob_70', {'crossover_prob': 0.70}),
    ('crossprob_100', {'crossover_prob': 1.0}),
    ('roulette', {'selection': 'roulette'}),
    ('two_point', {'crossover': 'two_point'}),
    ('mutation_005', {'mutation': 0.005})
]

def fitness(x, problem):
    if problem == 'OneMax':
        return int(np.sum(x))
    zeros = np.flatnonzero(x == 0)
    return int(zeros[0]) if zeros.size else len(x)

def choose(rng, cum_weights):
    return int(np.searchsorted(cum_weights, rng.random(), side='right'))

def crossover(a, b, kind, rng):
    d = len(a)
    if kind == 'one_point' or d < 3:
        point = rng.integers(1, d)
        return np.r_[b[:point], a[point:]], np.r_[a[:point], b[point:]]
    points = np.sort(rng.choice(np.arange(1, d), 2, replace=False))
    i, j = points
    x, y = a.copy(), b.copy()
    x[i:j], y[i:j] = b[i:j], a[i:j]
    return x, y

def ga(d, problem, seed=0, **params):
    rng = np.random.default_rng(seed)
    budget = 100 * d
    n = params['pop_size']
    assert 2 <= n <= budget
    population = rng.integers(0, 2, (n, d), dtype=np.uint8)
    scores = np.empty(n, dtype=int)
    history = np.empty(budget, dtype=int)
    best = 0
    for i in range(n):
        scores[i] = fitness(population[i], problem)
        best = max(best, scores[i])
        history[i] = best
    evaluations = n
    while evaluations < budget:
        elite_n = max(1, min(n - 1, int(n * params['elite'])))
        elite_ids = np.argsort(scores, kind='stable')[-elite_n:]
        next_pop = [population[i].copy() for i in elite_ids]
        next_scores = [int(scores[i]) for i in elite_ids]
        if params['selection'] == 'rank':
            ranks = np.empty(n, dtype=float)
            ranks[np.argsort(scores, kind='stable')] = np.arange(1, n + 1)
            weights = ranks
        else:
            weights = scores.astype(float) + 1.0
        cumulative = np.cumsum(weights / weights.sum())
        cumulative[-1] = 1.0
        while len(next_pop) < n and evaluations < budget:
            a = population[choose(rng, cumulative)]
            b = population[choose(rng, cumulative)]
            if rng.random() < params['crossover_prob']:
                children = crossover(a, b, params['crossover'], rng)
            else:
                children = a.copy(), b.copy()
            for child in children:
                if len(next_pop) >= n or evaluations >= budget:
                    break
                child ^= (rng.random(d) < params['mutation']).astype(np.uint8)
                value = fitness(child, problem)
                next_pop.append(child)
                next_scores.append(value)
                best = max(best, value)
                history[evaluations] = best
                evaluations += 1
        if len(next_pop) == n:
            population = np.array(next_pop, dtype=np.uint8)
            scores = np.array(next_scores, dtype=int)
    assert evaluations == budget and len(history) == budget
    return history, int(best)

def main():
    rows, histories, individual = [], {}, []
    for problem in PROBLEMS:
        for d in DIMS:
            for name, changes in TESTS:
                params = BASE | changes
                curves, finals = [], []
                for seed in range(RUNS):
                    curve, final = ga(d, problem, seed=seed, **params)
                    assert len(curve) == 100 * d and np.all(np.diff(curve) >= 0)
                    curves.append(curve)
                    finals.append(final)
                    individual.append(dict(problem=problem, dimension=d, configuration=name, run=seed + 1, result=final))
                values = np.array(finals)
                histories[(problem, d, name)] = np.mean(curves, axis=0)
                rows.append(dict(problem=problem, dimension=d, configuration=name,
                                 best=int(values.max()), worst=int(values.min()),
                                 mean=float(values.mean()), median=float(np.median(values)),
                                 std=float(values.std()), success_rate=float(np.mean(values == d)),
                                 evaluations=100 * d, **params))
                print(f'{problem} {d}D {name}: {values.mean():.2f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / 'statistiky.csv', index=False)
    pd.DataFrame(individual).to_csv(OUTPUT / 'jednotlive_behy.csv', index=False)
    winners = df.sort_values(['mean', 'std'], ascending=[False, True]).groupby(['problem', 'dimension'], sort=False).head(1)
    winners.to_csv(OUTPUT / 'nejlepsi_konfigurace.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, problem in enumerate(PROBLEMS):
        for j, d in enumerate(DIMS):
            ax = axes[i, j]
            for name, _ in TESTS:
                ax.plot(np.arange(1, 100*d + 1), histories[(problem, d, name)], label=name, linewidth=1.25)
            ax.set(title=f'{problem} – {d}D', xlabel='Počet evaluací', ylabel='Průměrná nejlepší fitness', ylim=(0, d + 1))
            ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(OUTPUT / 'konvergence_vse.png', dpi=150)
    plt.close(fig)
    for problem in PROBLEMS:
        for d in DIMS:
            fig, ax = plt.subplots(figsize=(9, 5))
            for name, _ in TESTS:
                ax.plot(np.arange(1, 100*d + 1), histories[(problem, d, name)], label=name)
            ax.set(title=f'{problem} – {d}D, průměr z 10 běhů', xlabel='Počet evaluací', ylabel='Průměrná nejlepší fitness', ylim=(0, d + 1))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(OUTPUT / f'konvergence_{problem}_{d}D.png', dpi=150)
            plt.close(fig)
    print('Hotovo:', OUTPUT)

if __name__ == '__main__':
    main()
