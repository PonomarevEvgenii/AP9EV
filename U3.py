from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent / 'U3_results'
OUTPUT.mkdir(exist_ok=True)
RUNS = 10
D = 10
BUDGET = 10000
NP = 30
FUNCTIONS = {'Sphere': 5.12, 'Rosenbrock': 2.048, 'Schwefel': 500.0}
VARIANTS = ['Rand/1', 'Best/1', 'jDE']


def fitness(x, name):
    if name == 'Sphere':
        return float(np.sum(x ** 2))
    if name == 'Rosenbrock':
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def de(name, variant, seed=0):
    rng = np.random.default_rng(seed)
    bound = FUNCTIONS[name]
    pop = rng.uniform(-bound, bound, (NP, D))
    scores = np.array([fitness(x, name) for x in pop])
    F = np.full(NP, 0.5)
    CR = np.full(NP, 0.9 if variant == 'jDE' else 0.8)
    history = np.empty(BUDGET)
    best = np.inf
    for i in range(NP):
        best = min(best, scores[i])
        history[i] = best
    evaluations = NP
    while evaluations < BUDGET:
        for i in range(NP):
            if evaluations >= BUDGET:
                break
            f, cr = F[i], CR[i]
            if variant == 'jDE':
                if rng.random() < 0.1:
                    f = 0.1 + rng.random() * 0.8
                if rng.random() < 0.1:
                    cr = rng.random()
            others = [j for j in range(NP) if j != i]
            r1, r2, r3 = rng.choice(others, 3, replace=False)
            if variant == 'Best/1':
                v = pop[np.argmin(scores)] + f * (pop[r1] - pop[r2])
            else:
                v = pop[r1] + f * (pop[r2] - pop[r3])
            v = np.clip(v, -bound, bound)
            mask = rng.random(D) < cr
            mask[rng.integers(D)] = True
            u = np.where(mask, v, pop[i])
            value = fitness(u, name)
            if value <= scores[i]:
                pop[i] = u
                scores[i] = value
                F[i], CR[i] = f, cr
            best = min(best, value)
            history[evaluations] = best
            evaluations += 1
    return history, best


def main():
    rows, histories, individual = [], {}, []
    for name in FUNCTIONS:
        for variant in VARIANTS:
            curves, finals = [], []
            for seed in range(RUNS):
                curve, final = de(name, variant, seed=seed)
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
            ax.plot(np.arange(1, BUDGET + 1), np.maximum(histories[(name, variant)], 1e-100), label=variant)
        ax.set(title=f'{name} – {D}D, průměr z {RUNS} běhů', xlabel='Počet evaluací', ylabel='Průměrná nejlepší hodnota účelové funkce', yscale='log')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / f'konvergence_{name}.png', dpi=150)
        plt.close(fig)
    print('Hotovo:', OUTPUT)


if __name__ == '__main__':
    main()
