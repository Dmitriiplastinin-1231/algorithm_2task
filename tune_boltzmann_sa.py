"""
Экспериментальный подбор параметров алгоритма имитации отжига с модификацией Больцмана.

Для каждого параметра выполняется серия запусков при фиксированных остальных параметрах.
По результатам строится и сохраняется график зависимости качества решения от значения параметра.

Параметры, которые перебираются:
  - restarts         (количество рестартов)
  - steps_per_restart (число шагов на один рестарт)
  - t0_multiplier    (множитель начальной температуры T₀)
  - min_temperature  (минимальная температура, нижний порог)

График температурного расписания Больцмана: T(t) = T₀ / ln(t + 2).

Графики сохраняются в папку results_boltzmann_sa/ рядом со скриптом.
"""

import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphClass import Graph
from otjig import simulated_annealing

# ---------------------------------------------------------------------------
# Настройки эксперимента
# ---------------------------------------------------------------------------
STP_FILE = os.path.join(os.path.dirname(__file__), "berlin52.stp")
RUNS_PER_VALUE = 5        # запусков (разных seed) для каждого значения параметра
BASE_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results_boltzmann_sa")

# Базовые (фиксированные) значения параметров
BASE_PARAMS = {
    "restarts": 8,
    "steps_per_restart": 5000,
    "t0_multiplier": 1.0,
    "min_temperature": 1e-4,
}

# Диапазоны перебираемых значений
PARAM_GRIDS = {
    "restarts":          [2, 4, 6, 8, 10, 12, 16],
    "steps_per_restart": [500, 1000, 2000, 5000, 10000, 20000],
    "t0_multiplier":     [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "min_temperature":   [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def run_experiment(graph, params, seed):
    """Запускает один прогон отжига Больцмана и возвращает длину лучшего маршрута."""
    _, length = simulated_annealing(
        graph,
        restarts=params["restarts"],
        steps_per_restart=params["steps_per_restart"],
        seed=seed,
        acceptance_mode="boltzmann",
        t0_multiplier=params["t0_multiplier"],
        min_temperature=params["min_temperature"],
    )
    return length


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
        print(f"  {param_name}={val!r:>12}  mean={means[-1]:.2f}  std={stds[-1]:.2f}")
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

    best_idx = values.index(best_value)
    ax.axvline(best_idx, color="red", linestyle="--", linewidth=1.5,
               label=f"Лучшее: {best_value}")
    ax.scatter([best_idx], [means[best_idx]], color="red", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Длина лучшего маршрута", fontsize=12)
    ax.set_title(f"Отжиг Больцмана — подбор параметра «{param_name}»", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(output_dir, f"boltzmann_sa_{param_name}.png")
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

    recommended = {}

    for param_name, values in PARAM_GRIDS.items():
        print(f"\nПеребор параметра: {param_name}")
        vals, means, stds = sweep_param(graph, param_name, values)

        finite_pairs = [(m, v) for m, v in zip(means, vals) if not math.isnan(m)]
        if not finite_pairs:
            print(f"  Нет допустимых результатов для {param_name}, пропускаем.")
            continue
        best_mean, best_val = min(finite_pairs)
        recommended[param_name] = best_val
        print(f"  Лучшее значение: {param_name}={best_val}  (mean={best_mean:.2f})")

        plot_and_save(param_name, vals, means, stds, OUTPUT_DIR, best_val)

    print("\n=== Рекомендуемые параметры (отжиг Больцмана) ===")
    for k, v in recommended.items():
        print(f"  {k} = {v}")
    print(f"\nГрафики сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
