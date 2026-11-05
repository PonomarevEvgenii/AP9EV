from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent / 'U4_results'
OUTPUT.mkdir(exist_ok=True)
RUNS = 10
D = 10
MAXFES = 10000
NP = 30
FUNCTIONS = {'Sphere': 5.12, 'Rosenbrock': 2.048, 'Schwefel': 500.0}
VARIANTS = {
    'w 0.8→0.3, hvězda': {'w': None, 'c': 2.0, 'topology': 'star'},
    'w 0.7, hvězda': {'w': 0.7, 'c': 1.49618, 'topology': 'star'},
    'w 0.7, kruh': {'w': 0.7, 'c': 1.49618, 'topology': 'ring'}
}


def fitness(x, name):
    if name == 'Sphere':
        return float(np.sum(x ** 2))
    if name == 'Rosenbrock':
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def pso(name, w, c, topology, seed=0):
    rng = np.random.default_rng(seed)
    bound = FUNCTIONS[name]
    vmax = 0.2 * 2 * bound
    x = rng.uniform(-bound, bound, (NP, D))
    v = rng.uniform(-vmax, vmax, (NP, D))
    scores = np.array([fitness(p, name) for p in x])
    pbest = x.copy()
    pbest_scores = scores.copy()
    history = np.empty(MAXFES)
    best = np.inf
    for i in range(NP):
        best = min(best, scores[i])
        history[i] = best
    fes = NP
    while fes < MAXFES:
        inertia = 0.8 - 0.5 * fes / MAXFES if w is None else w
        for i in range(NP):
            if fes >= MAXFES:
                break
            if topology == 'star':
                leader = pbest[np.argmin(pbest_scores)]
            else:
                neighbours = [(i - 1) % NP, i, (i + 1) % NP]
                leader = pbest[neighbours[np.argmin(pbest_scores[neighbours])]]
            v[i] = inertia * v[i] + c * rng.random(D) * (pbest[i] - x[i]) + c * rng.random(D) * (leader - x[i])
            v[i] = np.clip(v[i], -vmax, vmax)
            x[i] = x[i] + v[i]
            if np.any(np.abs(x[i]) > bound):
                continue
            value = fitness(x[i], name)
            if value < pbest_scores[i]:
                pbest[i] = x[i].copy()
                pbest_scores[i] = value
            best = min(best, value)
            history[fes] = best
            fes += 1
    return history, best


def main():
    rows, histories, individual = [], {}, []
    for name in FUNCTIONS:
        for variant, params in VARIANTS.items():
            curves, finals = [], []
            for seed in range(RUNS):
                curve, final = pso(name, seed=seed, **params)
                curves.append(curve)
                finals.append(final)
                individual.append(dict(function=name, variant=variant, run=seed + 1, result=final))
            values = np.array(finals)
            histories[(name, variant)] = np.mean(curves, axis=0)
            rows.append(dict(function=name, variant=variant, best=values.min(), worst=values.max(),
                             mean=values.mean(), median=np.median(values), std=values.std()))
            print(f'{name} {variant}: {values.mean():.6g}', flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / 'statistiky.csv', index=False)
    pd.DataFrame(individual).to_csv(OUTPUT / 'jednotlive_behy.csv', index=False)
    for name in FUNCTIONS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for variant in VARIANTS:
            ax.plot(np.arange(1, MAXFES + 1), np.maximum(histories[(name, variant)], 1e-100), label=variant)
        ax.set(title=f'{name} – {D}D, průměr z {RUNS} běhů', xlabel='Počet evaluací', ylabel='Průměrná nejlepší hodnota účelové funkce', yscale='log')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / f'konvergence_{name}.png', dpi=150)
        plt.close(fig)
    print('Hotovo:', OUTPUT)


if __name__ == '__main__':
    main()
