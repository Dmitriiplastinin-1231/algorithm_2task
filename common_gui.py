import queue
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ant_gui import AntColonyGUI
from graphClass import Graph
from otjig import simulated_annealing


def path_edges(path):
    if not path or len(path) < 2:
        return []
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


class AnnealingGUI:
    def __init__(self, root):
        self.root = root

        self.graph = None
        self.graph_nx = None
        self.pos = None

        self.events = queue.Queue()
        self.worker = None

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self):
        control = ttk.Frame(self.root, padding=10)
        control.pack(side=tk.TOP, fill=tk.X)

        self.file_var = tk.StringVar(value="")
        self.restarts_var = tk.StringVar(value="8")
        self.steps_var = tk.StringVar(value="")
        self.seed_var = tk.StringVar(value="42")

        ttk.Label(control, text="Файл графа (.stp):").grid(row=0, column=0, sticky="w")
        ttk.Entry(control, textvariable=self.file_var, width=45).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(control, text="Выбрать", command=self.choose_file).grid(row=0, column=2, padx=5)
        ttk.Button(control, text="Загрузить", command=self.load_graph).grid(row=0, column=3, padx=5)

        ttk.Label(control, text="Рестарты").grid(row=1, column=0, sticky="w")
        ttk.Entry(control, textvariable=self.restarts_var, width=8).grid(row=1, column=1, sticky="w")

        ttk.Label(control, text="Шагов на рестарт (пусто = авто)").grid(row=1, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(control, textvariable=self.steps_var, width=12).grid(row=1, column=3, sticky="w", padx=(5, 0))

        ttk.Label(control, text="seed").grid(row=1, column=4, sticky="w", padx=(12, 0))
        ttk.Entry(control, textvariable=self.seed_var, width=8).grid(row=1, column=5, sticky="w", padx=(5, 0))

        self.start_btn = ttk.Button(control, text="Запустить имитацию отжига", command=self.run_solver)
        self.start_btn.grid(row=2, column=0, columnspan=3, sticky="we", pady=(8, 0))

        self.status_var = tk.StringVar(value="Загрузите граф")
        ttk.Label(control, textvariable=self.status_var).grid(row=2, column=3, columnspan=3, sticky="w", padx=5)

        control.columnconfigure(1, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Выберите STP-файл",
            filetypes=[("STP files", "*.stp"), ("All files", "*.*")],
        )
        if path:
            self.file_var.set(path)

    def load_graph(self):
        file_name = self.file_var.get().strip()
        if not file_name:
            messagebox.showerror("Ошибка", "Укажите путь к .stp файлу.")
            return
        try:
            self.graph = Graph.load_from_stp(file_name)
            seed_text = self.seed_var.get().strip()
            layout_seed = int(seed_text) if seed_text else 42
            self._build_visual_graph(seed=layout_seed)
            self.status_var.set(f"Граф загружен: {self.graph.num_nodes} вершин")
            self._draw_state()
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", str(exc))

    def _build_visual_graph(self, max_edges=5000, seed=42):
        graph_nx = nx.Graph()
        for u in range(self.graph.num_nodes):
            graph_nx.add_node(u)
        for u in range(self.graph.num_nodes):
            for v, w in self.graph.adj[u]:
                if u < v:
                    graph_nx.add_edge(u, v, weight=w)

        if graph_nx.number_of_edges() > max_edges:
            rnd = random.Random(seed)
            sampled_edges = rnd.sample(list(graph_nx.edges()), max_edges)
            sampled = nx.Graph()
            sampled.add_nodes_from(graph_nx.nodes())
            sampled.add_edges_from(sampled_edges)
            self.graph_nx = sampled
        else:
            self.graph_nx = graph_nx

        n = self.graph_nx.number_of_nodes()
        if n > 300:
            self.pos = nx.circular_layout(self.graph_nx)
        elif n > 120:
            self.pos = nx.spring_layout(self.graph_nx, seed=seed, k=0.15, iterations=35)
        else:
            self.pos = nx.spring_layout(self.graph_nx, seed=seed)

    def _draw_state(self, best_path=None, best_length=None):
        self.ax.clear()
        if self.graph_nx is None:
            self.ax.set_title("Граф не загружен")
            self.ax.axis("off")
            self.canvas.draw_idle()
            return

        nx.draw_networkx_nodes(
            self.graph_nx,
            self.pos,
            ax=self.ax,
            node_size=15 if self.graph.num_nodes > 150 else 40,
            node_color="lightblue",
            alpha=0.9,
        )
        nx.draw_networkx_edges(
            self.graph_nx,
            self.pos,
            ax=self.ax,
            edge_color="gray",
            alpha=0.25,
            width=0.6,
        )

        if best_path:
            nx.draw_networkx_edges(
                self.graph_nx,
                self.pos,
                ax=self.ax,
                edgelist=path_edges(best_path),
                edge_color="dodgerblue",
                width=2.0,
                alpha=0.95,
            )

        title = "Имитация отжига: визуализация"
        if best_length is not None and best_length != float("inf"):
            title += f" | лучшая длина {best_length:.2f}"
        self.ax.set_title(title)
        self.ax.axis("off")
        self.canvas.draw_idle()

    def run_solver(self):
        if self.worker and self.worker.is_alive():
            return
        if self.graph is None:
            self.load_graph()
            if self.graph is None:
                return

        try:
            restarts = int(self.restarts_var.get())
            steps_text = self.steps_var.get().strip()
            steps_per_restart = int(steps_text) if steps_text else None
            seed_text = self.seed_var.get().strip()
            seed = int(seed_text) if seed_text else None
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте числовые параметры.")
            return

        self.start_btn.configure(state="disabled")
        self.status_var.set("Расчёт запущен...")

        def worker():
            started = time.time()
            best_tour, best_length = simulated_annealing(
                self.graph,
                restarts=restarts,
                steps_per_restart=steps_per_restart,
                seed=seed,
            )
            elapsed = time.time() - started
            self.events.put(("done", best_tour, best_length, elapsed))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "done":
                    _, best_tour, best_length, elapsed = event
                    self.start_btn.configure(state="normal")
                    if not best_tour:
                        self.status_var.set(f"Завершено за {elapsed:.2f} c. Цикл не найден.")
                        self._draw_state()
                    else:
                        cycle = best_tour + [best_tour[0]]
                        self.status_var.set(
                            f"Готово за {elapsed:.2f} c. Лучшая длина: {best_length:.2f}"
                        )
                        self._draw_state(best_path=cycle, best_length=best_length)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_events)


class CommonAlgorithmsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Общий GUI для алгоритмов TSP")
        self.root.geometry("1300x820")

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)

        ant_tab = ttk.Frame(notebook)
        anneal_tab = ttk.Frame(notebook)

        notebook.add(ant_tab, text="Муравьиный алгоритм")
        notebook.add(anneal_tab, text="Имитация отжига")

        self.ant_gui = AntColonyGUI(ant_tab)
        self.anneal_gui = AnnealingGUI(anneal_tab)


def main():
    root = tk.Tk()
    CommonAlgorithmsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
