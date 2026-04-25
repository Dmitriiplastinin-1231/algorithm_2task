"""
Экспериментальный подбор параметров муравьиного алгоритма с модификацией «элитные муравьи».

Для каждого параметра выполняется серия запусков при фиксированных остальных параметрах.
По результатам строится и сохраняется график зависимости качества решения от значения параметра.

Параметры, которые перебираются:
  - ants        (количество муравьёв)
  - alpha       (влияние феромона)
  - beta        (влияние эвристики)
  - evaporation (скорость испарения феромона)
  - q           (количество феромона, откладываемого муравьём)
  - elite_ants  (количество «элитных» муравьёв — ключевой параметр модификации)

Графики сохраняются в папку results_elite/ рядом со скриптом.
"""

import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphClass import Graph
from ant_colony import AntColonyTSP

# ---------------------------------------------------------------------------
# Настройки эксперимента
# ---------------------------------------------------------------------------
STP_FILE = os.path.join(os.path.dirname(__file__), "berlin52.stp")
ITERATIONS = 100          # итераций на один запуск
RUNS_PER_VALUE = 5        # запусков (разных seed) для каждого значения параметра
BASE_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results_elite")

# Базовые (фиксированные) значения параметров
BASE_PARAMS = {
    "ants": 20,
    "alpha": 1.0,
    "beta": 3.0,
    "evaporation": 0.5,
    "q": 100.0,
    "elite_ants": 5,
}

# Диапазоны перебираемых значений
PARAM_GRIDS = {
    "ants":        [5, 10, 20, 30, 40, 50, 60],
    "alpha":       [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "beta":        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "evaporation": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    "q":           [10, 50, 100, 200, 500, 1000],
    "elite_ants":  [1, 2, 5, 10, 15, 20],
}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def run_experiment(graph, params, seed):
    """Запускает один прогон алгоритма с элитными муравьями и возвращает длину лучшего маршрута."""
    solver = AntColonyTSP(
        graph,
        ants=params["ants"],
        alpha=params["alpha"],
        beta=params["beta"],
        evaporation=params["evaporation"],
        q=params["q"],
        seed=seed,
        mode="elite",
        elite_ants=params["elite_ants"],
    )
    result = solver.solve(iterations=ITERATIONS)
    return result.best_length if result is not None else float("inf")


def sweep_param(graph, param_name, values):
    """
    Перебирает значения одного параметра, усредняет результат по нескольким seed.
    Возвращает списки (values, means, stds).
    """
    means = []
    stds = []
    for val in values:
        params = dict(BASE_PARAMS)
        params[param_name] = val
        lengths = [
            run_experiment(graph, params, seed=BASE_SEED + i)
            for i in range(RUNS_PER_VALUE)
        ]
        finite = [x for x in lengths if x < float("inf")]
        if finite:
            means.append(statistics.mean(finite))
            stds.append(statistics.stdev(finite) if len(finite) > 1 else 0.0)
        else:
            means.append(float("nan"))
            stds.append(0.0)
        print(f"  {param_name}={val:>8}  mean={means[-1]:.2f}  std={stds[-1]:.2f}")
    return values, means, stds


def plot_and_save(param_name, values, means, stds, output_dir, best_value):
    """Строит и сохраняет график для одного параметра."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = list(range(len(values)))
    ax.plot(x, means, marker="o", color="darkorange", linewidth=2, label="Среднее")
    ax.fill_between(
        x,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.25,
        color="darkorange",
        label="±std",
    )

    # Отмечаем лучшее значение
    best_idx = values.index(best_value)
    ax.axvline(best_idx, color="red", linestyle="--", linewidth=1.5,
               label=f"Лучшее: {best_value}")
    ax.scatter([best_idx], [means[best_idx]], color="red", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Длина лучшего маршрута", fontsize=12)
    ax.set_title(f"АСО с элитными муравьями — подбор параметра «{param_name}»", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(output_dir, f"elite_aco_{param_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → сохранено: {path}")


# ---------------------------------------------------------------------------
# Основная программа
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Загрузка графа из {STP_FILE} …")
    graph = Graph.load_from_stp(STP_FILE)
    print(f"Граф: {graph.num_nodes} вершин")

    for param_name, values in PARAM_GRIDS.items():
        print(f"\nПеребор параметра: {param_name}")
        vals, means, stds = sweep_param(graph, param_name, values)

        # Выбираем значение с минимальной средней длиной
        finite_pairs = [(m, v) for m, v in zip(means, vals) if m == m]  # исключаем nan
        if not finite_pairs:
            print(f"  Нет допустимых результатов для {param_name}, пропускаем.")
            continue
        best_mean, best_val = min(finite_pairs)
        print(f"  Лучшее значение: {param_name}={best_val}  (mean={best_mean:.2f})")

        plot_and_save(param_name, vals, means, stds, OUTPUT_DIR, best_val)

    print("\nГотово. Графики сохранены в", OUTPUT_DIR)


if __name__ == "__main__":
    main()
