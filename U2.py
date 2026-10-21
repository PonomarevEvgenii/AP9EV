from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent / 'U2_results'
OUTPUT.mkdir(exist_ok=True)
RUNS = 10
D = 10
BUDGET = 10000
BITS = 32
BASE = dict(pop_size=50, elite=0.10, crossover_prob=0.9)
FUNCTIONS = {'Sphere': 5.12, 'Rosenbrock': 2.048, 'Schwefel': 500.0}
VARIANTS = [
    ('IEEE754', {'kind': 'ieee', 'mutation': 0.005}),
    ('FixedPoint', {'kind': 'fixed', 'mutation': 0.005}),
    ('BCD', {'kind': 'bcd', 'mutation': 0.005}),
    ('Real_N_s0.01_p0.1', {'kind': 'gauss', 'sigma': 0.01, 'mutation': 0.1}),
    ('Real_N_s0.05_p0.1', {'kind': 'gauss', 'sigma': 0.05, 'mutation': 0.1}),
    ('Real_N_s0.1_p0.1', {'kind': 'gauss', 'sigma': 0.1, 'mutation': 0.1}),
    ('Real_N_s0.01_p0.3', {'kind': 'gauss', 'sigma': 0.01, 'mutation': 0.3}),
    ('Real_uniform_p0.1', {'kind': 'uniform', 'mutation': 0.1})
]
np.seterr(all='ignore')
SHIFTS = np.arange(BITS - 1, -1, -1, dtype=np.uint64)


def fitness(x, name):
    if name == 'Sphere':
        return float(np.sum(x ** 2))
    if name == 'Rosenbrock':
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def int_digits(bound):
    return len(str(int(bound)))


def frac_bits(bound):
    return BITS - int(np.ceil(np.log2(bound))) - 1


def to_bits(u):
    return ((u.astype(np.uint64)[:, None] >> SHIFTS) & 1).astype(np.uint8).ravel()


def from_bits(bits):
    return (bits.reshape(D, BITS).astype(np.uint64) << SHIFTS).sum(axis=1).astype(np.uint32)


def encode(x, kind, bound):
    if kind == 'ieee':
        return to_bits(x.astype(np.float32).view(np.uint32))
    if kind == 'fixed':
        return to_bits((np.round(x * 2.0 ** frac_bits(bound)).astype(np.int64) & 0xFFFFFFFF).astype(np.uint32))
    decimals = 7 - int_digits(bound)
    value = np.round(np.abs(x) * 10 ** decimals).astype(np.int64)
    u = np.where(x < 0, 1, 0).astype(np.int64) << 28
    for i in range(7):
        u = u | (((value // 10 ** (6 - i)) % 10) << (24 - 4 * i))
    return to_bits(u.astype(np.uint32))


def decode(bits, kind, bound):
    u = from_bits(bits)
    if kind == 'ieee':
        x = np.nan_to_num(u.view(np.float32).astype(float), nan=0.0, posinf=bound, neginf=-bound)
    elif kind == 'fixed':
        x = u.view(np.int32) / 2.0 ** frac_bits(bound)
    else:
        nibbles = (u[:, None] >> np.arange(28, -1, -4, dtype=np.uint32)) & 15
        digits = np.minimum(nibbles[:, 1:], 9)
        value = (digits * 10.0 ** np.arange(6, -1, -1)).sum(axis=1) / 10 ** (7 - int_digits(bound))
        x = np.where(nibbles[:, 0] % 2 == 1, -value, value)
    return np.clip(x, -bound, bound)


def choose(rng, cum_weights):
    return int(np.searchsorted(cum_weights, rng.random(), side='right'))


def ga(name, seed=0, **params):
    rng = np.random.default_rng(seed)
    bound = FUNCTIONS[name]
    kind = params['kind']
    binary = kind in ('ieee', 'fixed', 'bcd')
    n = params['pop_size']
    start = rng.uniform(-bound, bound, (n, D))
    if binary:
        population = np.array([encode(x, kind, bound) for x in start])
    else:
        population = start
    length = population.shape[1]
    scores = np.empty(n)
    history = np.empty(BUDGET)
    best = np.inf
    for i in range(n):
        x = decode(population[i], kind, bound) if binary else population[i]
        scores[i] = fitness(x, name)
        best = min(best, scores[i])
        history[i] = best
    evaluations = n
    elite_n = max(1, int(n * params['elite']))
    ranks = np.arange(n, 0, -1, dtype=float)
    cumulative = np.cumsum(ranks / ranks.sum())
    cumulative[-1] = 1.0
    while evaluations < BUDGET:
        order = np.argsort(scores, kind='stable')
        next_pop = [population[i].copy() for i in order[:elite_n]]
        next_scores = [scores[i] for i in order[:elite_n]]
        while len(next_pop) < n and evaluations < BUDGET:
            a = population[order[choose(rng, cumulative)]]
            b = population[order[choose(rng, cumulative)]]
            if rng.random() < params['crossover_prob']:
                point = rng.integers(1, length)
                children = np.r_[a[:point], b[point:]], np.r_[b[:point], a[point:]]
            else:
                children = a.copy(), b.copy()
            for child in children:
                if len(next_pop) >= n or evaluations >= BUDGET:
                    break
                mask = rng.random(length) < params['mutation']
                if binary:
                    child[mask] ^= 1
                    x = decode(child, kind, bound)
                elif kind == 'gauss':
                    child[mask] += rng.normal(0, params['sigma'] * 2 * bound, mask.sum())
                    child = np.clip(child, -bound, bound)
                    x = child
                else:
                    child[mask] = rng.uniform(-bound, bound, mask.sum())
                    x = child
                value = fitness(x, name)
                next_pop.append(child)
                next_scores.append(value)
                best = min(best, value)
                history[evaluations] = best
                evaluations += 1
        if len(next_pop) == n:
            population = np.array(next_pop)
            scores = np.array(next_scores)
    return history, best


def main():
    test = np.array([-4.123456, 0.0, 3.5, -0.000123, 5.12, 1.0, -2.25, 0.75, 4.999999, -5.12])
    for kind in ('ieee', 'fixed', 'bcd'):
        assert np.allclose(decode(encode(test, kind, 5.12), kind, 5.12), test, atol=1e-5)
    rows, histories, individual = [], {}, []
    for name in FUNCTIONS:
        for variant, changes in VARIANTS:
            params = BASE | changes
            curves, finals = [], []
            for seed in range(RUNS):
                curve, final = ga(name, seed=seed, **params)
                curves.append(curve)
                finals.append(final)
                individual.append(dict(function=name, variant=variant, run=seed + 1, result=final))
            values = np.array(finals)
            histories[(name, variant)] = np.mean(curves, axis=0)
            rows.append(dict(function=name, variant=variant, best=values.min(), worst=values.max(),
                             mean=values.mean(), median=np.median(values), std=values.std(), **params))
            print(f'{name} {variant}: {values.mean():.6g}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / 'statistiky.csv', index=False)
    pd.DataFrame(individual).to_csv(OUTPUT / 'jednotlive_behy.csv', index=False)
    df.sort_values('mean').groupby('function', sort=False).head(1).to_csv(OUTPUT / 'nejlepsi_varianty.csv', index=False)
    for name in FUNCTIONS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for variant, _ in VARIANTS:
            ax.plot(np.arange(1, BUDGET + 1), np.maximum(histories[(name, variant)], 1e-100), label=variant)
        ax.set(title=f'{name} – {D}D, průměr z {RUNS} běhů', xlabel='Počet evaluací', ylabel='Průměrná nejlepší hodnota účelové funkce', yscale='log')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(OUTPUT / f'konvergence_{name}.png', dpi=150)
        plt.close(fig)
    ind = pd.DataFrame(individual)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, name in zip(axes, FUNCTIONS):
        data = [np.maximum(ind[(ind.function == name) & (ind.variant == v)].result.values, 1e-100) for v, _ in VARIANTS]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(VARIANTS) + 1), [v for v, _ in VARIANTS], rotation=60, ha='right', fontsize=8)
        ax.set(title=f'{name} – {D}D', ylabel='Výsledná hodnota účelové funkce', yscale='log')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'boxplot_vysledky.png', dpi=150)
    plt.close(fig)
    print('Hotovo:', OUTPUT)


if __name__ == '__main__':
    main()
