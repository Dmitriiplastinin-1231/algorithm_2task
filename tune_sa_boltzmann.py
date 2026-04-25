"""
Экспериментальный подбор параметров алгоритма имитации отжига (Больцмановский отжиг).

Программа перебирает значения ключевых параметров алгоритма, запускает его несколько
раз для каждого значения и строит графики зависимости длины маршрута и времени работы
от величины каждого параметра.

Отличие от классического отжига: температура убывает по закону Больцмана
    T(step) = T0 / ln(step + 2),
что обеспечивает более медленное охлаждение и тщательный поиск.
"""

import math
import time
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphClass import Graph
from otjig import simulated_annealing

# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────
DATA_FILE = "berlin52.stp"          # Используемый датасет
TRIALS = 5                          # Число запусков для каждого значения параметра
BASE_SEED = 42                      # Базовый seed (к нему прибавляем номер trial)
ACCEPTANCE_MODE = "boltzmann"       # Режим отжига — Больцмановский
OUTPUT_DIR = "."                    # Папка для сохранения графиков

# Диапазоны исследуемых параметров
RESTARTS_VALUES = [1, 2, 4, 6, 8, 10, 12, 16]
STEPS_VALUES = [500, 1000, 2000, 5000, 8000, 10000, 15000, 20000]

# Фиксированные параметры при исследовании каждого из двух параметров
DEFAULT_RESTARTS = 8
DEFAULT_STEPS = None                # None → рассчитывается автоматически в otjig.py


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────

def run_trial(graph, restarts, steps_per_restart, seed):
    """Один запуск алгоритма; возвращает (длина_лучшего_маршрута, время_сек)."""
    t0 = time.perf_counter()
    _, best_len = simulated_annealing(
        graph,
        restarts=restarts,
        steps_per_restart=steps_per_restart,
        seed=seed,
        acceptance_mode=ACCEPTANCE_MODE,
    )
    elapsed = time.perf_counter() - t0
    return best_len, elapsed


def sweep_parameter(graph, param_name, param_values, fixed_restarts, fixed_steps):
    """
    Перебирает значения одного параметра, фиксируя остальные.
    Возвращает словарь:
        {
            "values": [...],
            "mean_length": [...],
            "std_length": [...],
            "mean_time": [...],
            "std_time": [...],
        }
    """
    mean_lengths, std_lengths = [], []
    mean_times, std_times = [], []

    for val in param_values:
        restarts = val if param_name == "restarts" else fixed_restarts
        steps    = val if param_name == "steps"    else fixed_steps

        lengths, times = [], []
        for t in range(TRIALS):
            length, elapsed = run_trial(graph, restarts, steps, seed=BASE_SEED + t)
            lengths.append(length)
            times.append(elapsed)

        n = len(lengths)
        m_l = sum(lengths) / n
        m_t = sum(times) / n
        s_l = math.sqrt(sum((x - m_l) ** 2 for x in lengths) / max(1, n - 1))
        s_t = math.sqrt(sum((x - m_t) ** 2 for x in times)   / max(1, n - 1))

        mean_lengths.append(m_l)
        std_lengths.append(s_l)
        mean_times.append(m_t)
        std_times.append(s_t)

        best = min(lengths)
        print(f"  {param_name}={val:>8}  лучшая длина={best:.1f}  "
              f"средняя={m_l:.1f}±{s_l:.1f}  время={m_t:.2f}±{s_t:.2f} с")

    return {
        "values": list(param_values),
        "mean_length": mean_lengths,
        "std_length": std_lengths,
        "mean_time": mean_times,
        "std_time": std_times,
    }


def plot_results(results, param_name, xlabel, title_suffix, filename_prefix):
    """Строит и сохраняет два графика: длина маршрута и время работы."""

    x = results["values"]
    ml = results["mean_length"]
    sl = results["std_length"]
    mt = results["mean_time"]
    st = results["std_time"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Больцмановский отжиг — влияние параметра «{title_suffix}»\n"
        f"({TRIALS} запусков на точку, датасет {DATA_FILE})",
        fontsize=12,
    )

    # ── Длина маршрута ──
    ax = axes[0]
    ax.errorbar(x, ml, yerr=sl, fmt="o-", color="mediumseagreen",
                ecolor="lightgreen", elinewidth=2, capsize=4,
                linewidth=1.8, markersize=6, label="Средняя длина ± σ")
    best_idx = ml.index(min(ml))
    ax.axvline(x=x[best_idx], color="red", linestyle="--", linewidth=1.2,
               label=f"Лучшее: {param_name}={x[best_idx]}")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Длина маршрута", fontsize=11)
    ax.set_title("Длина лучшего найденного маршрута", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ── Время работы ──
    ax = axes[1]
    ax.errorbar(x, mt, yerr=st, fmt="s-", color="mediumpurple",
                ecolor="thistle", elinewidth=2, capsize=4,
                linewidth=1.8, markersize=6, label="Среднее время ± σ")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Время (секунды)", fontsize=11)
    ax.set_title("Время работы алгоритма", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → График сохранён: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Главная программа
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Загрузка графа из {DATA_FILE} …")
    graph = Graph.load_from_stp(DATA_FILE)
    print(f"  Вершин: {graph.num_nodes}\n")

    # ── 1. Исследование параметра restarts ──────────────────────────────────
    print("=" * 60)
    print("Параметр: restarts  (steps_per_restart фиксировано = авто)")
    print("=" * 60)
    res_restarts = sweep_parameter(
        graph,
        param_name="restarts",
        param_values=RESTARTS_VALUES,
        fixed_restarts=DEFAULT_RESTARTS,
        fixed_steps=DEFAULT_STEPS,
    )
    plot_results(
        res_restarts,
        param_name="restarts",
        xlabel="Число рестартов (restarts)",
        title_suffix="restarts",
        filename_prefix="boltzmann_sa_restarts",
    )

    # ── 2. Исследование параметра steps_per_restart ──────────────────────────
    print()
    print("=" * 60)
    print("Параметр: steps_per_restart  (restarts фиксировано =", DEFAULT_RESTARTS, ")")
    print("=" * 60)
    res_steps = sweep_parameter(
        graph,
        param_name="steps",
        param_values=STEPS_VALUES,
        fixed_restarts=DEFAULT_RESTARTS,
        fixed_steps=DEFAULT_STEPS,
    )
    plot_results(
        res_steps,
        param_name="steps",
        xlabel="Шагов на рестарт (steps_per_restart)",
        title_suffix="steps_per_restart",
        filename_prefix="boltzmann_sa_steps",
    )

    # ── Итоговые рекомендации ────────────────────────────────────────────────
    best_restarts = RESTARTS_VALUES[res_restarts["mean_length"].index(
        min(res_restarts["mean_length"]))]
    best_steps = STEPS_VALUES[res_steps["mean_length"].index(
        min(res_steps["mean_length"]))]

    print()
    print("=" * 60)
    print("ИТОГОВЫЕ РЕКОМЕНДАЦИИ (Больцмановский отжиг):")
    print(f"  Лучшее значение restarts        : {best_restarts}")
    print(f"  Лучшее значение steps_per_restart: {best_steps}")
    print("=" * 60)


if __name__ == "__main__":
    main()
