"""
БЛОК 4 — ИНТЕРФЕЙС (NeutronSimGUI).

Графический интерфейс симулятора на Matplotlib + блок запуска.
Соответствует БЛОКУ 4 и блоку «ЗАПУСК» исходного монолита NSPM MCNP 3000.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons, Widget
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from simulation_worker import SimulationWorker, ExecutionState

from material import MATERIALS
from physics import estimate_k_full, DEFAULT_MAX_MEMORY_BYTES


class NeutronSimGUI:
    def __init__(self, show=True):
        # ── цвета ──────────────────────────────────────────────────
        self.C = {
            "bg":       "#0D1117",
            "panel":    "#161B22",
            "border":   "#30363D",
            "text":     "#E6EDF3",
            "dim":      "#8B949E",
            "blue":     "#378ADD",
            "green":    "#3FB950",
            "orange":   "#F78166",
            "yellow":   "#E3B341",
            "purple":   "#BC8CFF",
            "reactor":  "#1C2B1C",
            "react_bd": "#3FB950",
        }

        self.worker = SimulationWorker()
        self._closed = False
        self.seed = None
        self.max_samples_per_gen = 40
        self.max_steps = 500
        self.max_memory_bytes = DEFAULT_MAX_MEMORY_BYTES
        self.result_radius = None
        self.result_config = None
        self.result       = None
        self.current_gen  = 0
        self.anim_running = False
        self.anim_obj     = None

        self._build_layout()
        self._build_controls()
        self._draw_empty_reactor()
        self._timer = self.fig.canvas.new_timer(interval=100)
        self._timer.add_callback(self._poll_worker)
        self._timer.start()
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        if show:
            plt.show()

    # ── компоновка окна ────────────────────────────────────────────
    def _build_layout(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor=self.C["bg"])
        self.fig.canvas.manager.set_window_title("NeutronMC — симулятор активной зоны")

        gs = GridSpec(3, 3, figure=self.fig,
                      left=0.01, right=0.99,
                      top=0.94,  bottom=0.28,
                      wspace=0.08, hspace=0.35)

        # главный холст — траектории
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_main.set_facecolor(self.C["bg"])
        self.ax_main.set_aspect("equal")

        # heatmap делений
        self.ax_heat = self.fig.add_subplot(gs[0, 1])
        self.ax_heat.set_facecolor(self.C["bg"])
        self.ax_heat.set_aspect("equal")
        self.ax_heat.set_title("Карта делений", color=self.C["dim"], fontsize=8)

        # сходимость k
        self.ax_k = self.fig.add_subplot(gs[1, 1])
        self.ax_k.set_facecolor(self.C["panel"])
        self.ax_k.set_title("k-эффективное по поколениям", color=self.C["dim"], fontsize=8)

        # гистограмма судеб нейтронов
        self.ax_bar = self.fig.add_subplot(gs[2, 1])
        self.ax_bar.set_facecolor(self.C["panel"])
        self.ax_bar.set_title("Баланс нейтронов", color=self.C["dim"], fontsize=8)

        # статус-панель
        self.ax_stat = self.fig.add_subplot(gs[:, 2])
        self.ax_stat.set_facecolor(self.C["panel"])
        self.ax_stat.axis("off")

        # стили осей
        for ax in [self.ax_k, self.ax_bar, self.ax_heat]:
            ax.tick_params(colors=self.C["dim"], labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(self.C["border"])

        self.fig.text(0.5, 0.975,
                      "NeutronMC  ·  Симулятор активной зоны  ·  Метод Монте-Карло",
                      ha="center", va="top",
                      color=self.C["text"], fontsize=11, fontweight="bold",
                      fontfamily="monospace")

    # ── виджеты управления ─────────────────────────────────────────
    def _build_controls(self):
        bg, tc, bc = self.C["panel"], self.C["text"], self.C["border"]

        def make_ax(rect):
            a = self.fig.add_axes(rect, facecolor=bg)
            a.tick_params(colors=tc, labelsize=8)
            for sp in a.spines.values():
                sp.set_edgecolor(bc)
            return a

        # слайдеры
        self.sl_radius = Slider(make_ax([0.08, 0.18, 0.20, 0.025]),
                                "Радиус (см)", 5, 60, valinit=25,
                                color=self.C["blue"])
        self.sl_n      = Slider(make_ax([0.08, 0.13, 0.20, 0.025]),
                                "Нейтронов",  50, 2000, valinit=150, valstep=50,
                                color=self.C["green"])
        self.sl_gen    = Slider(make_ax([0.08, 0.08, 0.20, 0.025]),
                                "Поколений",  10, 60,  valinit=30, valstep=5,
                                color=self.C["yellow"])

        for sl in [self.sl_radius, self.sl_n, self.sl_gen]:
            sl.label.set_color(self.C["dim"])
            sl.label.set_fontsize(8)
            sl.valtext.set_color(self.C["text"])

        # кнопки
        self.btn_run  = Button(make_ax([0.36, 0.10, 0.09, 0.05]),
                               "▶  Запустить",
                               color=self.C["blue"], hovercolor="#4A9FE8")
        self.btn_anim = Button(make_ax([0.36, 0.04, 0.09, 0.05]),
                               "⟳  Анимация",
                               color=self.C["panel"], hovercolor=self.C["border"])
        self.btn_prev = Button(make_ax([0.47, 0.04, 0.04, 0.05]),
                               "◀", color=self.C["panel"], hovercolor=self.C["border"])
        self.btn_next = Button(make_ax([0.52, 0.04, 0.04, 0.05]),
                               "▶", color=self.C["panel"], hovercolor=self.C["border"])

        for btn in [self.btn_run, self.btn_anim, self.btn_prev, self.btn_next]:
            btn.label.set_color(self.C["text"])
            btn.label.set_fontsize(9)

        self.btn_cancel = Button(make_ax([0.47, 0.10, 0.09, 0.05]), "Cancel",
                                 color=self.C["panel"], hovercolor=self.C["border"])
        self.btn_cancel.label.set_color(self.C["text"])
        self.btn_cancel.on_clicked(lambda event: self.worker.cancel())
        self.btn_cancel.set_active(False)

        # выбор материала
        ax_radio = make_ax([0.62, 0.04, 0.16, 0.14])
        ax_radio.set_title("Материал", color=self.C["dim"], fontsize=8, pad=4)
        self.radio = RadioButtons(ax_radio, list(MATERIALS.keys()), active=0)
        n_labels = len(MATERIALS)
        self.radio.set_label_props({
            "color":    [self.C["dim"]] * n_labels,
            "fontsize": [8] * n_labels
        })

        # подключение событий
        self.btn_run.on_clicked(self._on_run)
        self.btn_anim.on_clicked(self._on_anim)
        self.btn_prev.on_clicked(lambda e: self._step_gen(-1))
        self.btn_next.on_clicked(lambda e: self._step_gen(+1))

    # ── пустой реактор при старте ──────────────────────────────────
    def _draw_empty_reactor(self):
        r = self.sl_radius.val
        ax = self.ax_main
        ax.clear()
        ax.set_facecolor(self.C["bg"])
        ax.set_xlim(-r*1.4, r*1.4)
        ax.set_ylim(-r*1.4, r*1.4)
        ax.set_aspect("equal")
        ax.set_title("Траектории нейтронов (2D проекция)",
                     color=self.C["dim"], fontsize=8)

        # зона замедлителя
        mod = patches.Circle((0,0), r*1.15, color="#0A2010", zorder=1)
        ax.add_patch(mod)
        # активная зона
        core = patches.Circle((0,0), r, facecolor=self.C["reactor"],
                               linewidth=1.5, edgecolor=self.C["react_bd"], zorder=2)
        ax.add_patch(core)
        # топливные элементы (схема)
        for i in range(6):
            ang = i / 6 * 2 * np.pi
            fx, fy = np.cos(ang)*r*0.55, np.sin(ang)*r*0.55
            rod = patches.Circle((fx, fy), r*0.07,
                                  color="#7B4F15", zorder=3)
            ax.add_patch(rod)
        # центральный элемент
        ax.add_patch(patches.Circle((0,0), r*0.07, color="#7B4F15", zorder=3))

        ax.text(0, 0, "Нажмите\nЗапустить", ha="center", va="center",
                color=self.C["dim"], fontsize=8, zorder=5)
        ax.axis("off")

        # ИСПРАВЛЕНИЕ: Обновляем канвас перед показом, чтобы избежать залипания
        self.fig.canvas.flush_events()
        self.fig.canvas.draw_idle()

    # ── запуск симуляции ───────────────────────────────────────────
    @property
    def execution_state(self):
        return self.worker.state

    def _set_running_controls(self, running):
        for widget in (self.sl_radius, self.sl_n, self.sl_gen, self.radio,
                       self.btn_run, self.btn_anim, self.btn_prev, self.btn_next):
            Widget.set_active(widget, not running)
            widget.ax.set_alpha(0.5 if running else 1.0)
        self.btn_cancel.set_active(running)

    def _on_run(self, event):
        if self.worker.state == ExecutionState.RUNNING or self._closed:
            return
        if self.anim_obj:
            self.anim_obj.event_source.stop()
            self.anim_obj = None
        self.anim_running = False
        self.btn_anim.label.set_text("⟳  Анимация")
        try:
            radius = float(self.sl_radius.val)
            config = dict(n_gen=int(self.sl_gen.val), n_source=int(self.sl_n.val),
                          radius=radius, material=MATERIALS[self.radio.value_selected],
                          seed=self.seed, max_samples_per_gen=self.max_samples_per_gen,
                          max_steps=self.max_steps, max_memory_bytes=self.max_memory_bytes)
            self.result = None
            self.result_radius = radius
            self.result_config = config
            self.worker.start(estimate_k_full, **config)
        except Exception as exc:
            self.worker.state = ExecutionState.FAILED
            self._set_running_controls(False)
            self._update_status(f"{type(exc).__name__}: {exc}", self.C["orange"])
        else:
            self._set_running_controls(True)
            self._update_status("RUNNING — calculating…", self.C["yellow"])
        self.fig.canvas.draw_idle()

    def _poll_worker(self):
        message = self.worker.poll()
        if message is None or self._closed:
            return
        state, payload = message
        self._set_running_controls(False)
        if state == ExecutionState.COMPLETED:
            self.result = payload
            self.current_gen = max(0, len(payload["k_history"]) - 1)
            try:
                self._render_all(self.result_radius)
            except Exception as exc:
                self.worker.state = ExecutionState.FAILED
                self.result = None
                self._update_status(f"Render failed: {type(exc).__name__}: {exc}", self.C["orange"])
        else:
            self.result = None
            self._update_status(payload if state == ExecutionState.FAILED else "CANCELLED",
                                self.C["orange"])
        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        self._closed = True
        self._timer.stop()
        self.worker.cancel()
        if self.anim_obj and self.anim_obj.event_source:
            self.anim_obj.event_source.stop()
        # The non-daemon worker exits at the next generation checkpoint.
        # No blocking join on the GUI thread, and no unsafe termination.

    # ── рендер всех панелей ────────────────────────────────────────
    def _render_all(self, radius=None):
        if self.result is None:
            return
        if not self.result["k_history"]:
            self._update_status_full()
            self.fig.canvas.draw_idle()
            return
        if radius is None:
            radius = self.result_radius or float(self.sl_radius.val)

        self.current_gen = max(0, min(self.current_gen, len(self.result["k_history"]) - 1))
        self._draw_trajectories(radius)
        self._draw_k_chart()
        self._draw_heatmap(radius)
        self._draw_bar()
        self._update_status_full()
        self.fig.canvas.draw_idle()

    # ── траектории ─────────────────────────────────────────────────
    def _draw_trajectories(self, radius):
        ax = self.ax_main
        ax.clear()
        ax.set_facecolor(self.C["bg"])
        ax.set_xlim(-radius*1.4, radius*1.4)
        ax.set_ylim(-radius*1.4, radius*1.4)
        ax.set_aspect("equal")
        ax.axis("off")

        if not self.result or not self.result["k_history"]:
            return
        gen_idx = min(self.current_gen, len(self.result["k_history"])-1)
        gen_idx = max(0, gen_idx)
        title = f"Поколение {gen_idx+1}/{len(self.result['k_history'])}  |  " \
                f"k = {self.result['k_history'][gen_idx]:.4f}"
        ax.set_title(title, color=self.C["text"], fontsize=9, pad=4)

        # замедлитель
        ax.add_patch(patches.Circle((0,0), radius*1.15,
                                    color="#0A2010", zorder=1))
        # активная зона
        ax.add_patch(patches.Circle((0,0), radius,
                                    facecolor=self.C["reactor"],
                                    linewidth=1.5,
                                    edgecolor=self.C["react_bd"], zorder=2))
        # топливные элементы
        for i in range(6):
            ang = i / 6 * 2 * np.pi
            fx, fy = np.cos(ang)*radius*0.55, np.sin(ang)*radius*0.55
            ax.add_patch(patches.Circle((fx, fy), radius*0.07,
                                         color="#7B4F15", zorder=3))
        ax.add_patch(patches.Circle((0,0), radius*0.07,
                                     color="#7B4F15", zorder=3))

        # траектории нейтронов
        fate_colors = {
            "fission":    self.C["orange"],
            "absorption": self.C["purple"],
            "leak":       self.C["blue"],
            "unfinished": self.C["dim"],
        }
        frames = self.result["path_frames"]
        paths = frames[gen_idx] if gen_idx < len(frames) else []
        for path, fate in paths:
            if len(path) < 2:
                continue
            pts = np.asarray(path)
            color = fate_colors.get(fate, self.C["dim"])
            ax.plot(pts[:,0], pts[:,1],
                    color=color, alpha=0.45, linewidth=0.7, zorder=4)
            ax.scatter(pts[-1,0], pts[-1,1],
                       s=10, color=color, alpha=0.8, zorder=5)

        # легенда
        legend_items = [
            ("деление",    self.C["orange"]),
            ("поглощение", self.C["purple"]),
            ("утечка",     self.C["blue"]),
            ("не завершены", self.C["dim"]),
        ]
        for i, (label, color) in enumerate(legend_items):
            ax.plot([], [], color=color, linewidth=1.5,
                    label=label, alpha=0.8)
        ax.legend(loc="lower left", fontsize=7,
                  facecolor=self.C["panel"],
                  edgecolor=self.C["border"],
                  labelcolor=self.C["text"],
                  framealpha=0.8)

    # ── график k ──────────────────────────────────────────────────
    def _draw_k_chart(self):
        ax = self.ax_k
        ax.clear()
        ax.set_facecolor(self.C["panel"])
        ax.set_title("k-эффективное по поколениям",
                     color=self.C["dim"], fontsize=8)

        k_hist = self.result["k_history"]
        gens   = list(range(1, len(k_hist)+1))

        ax.plot(gens, k_hist, color=self.C["blue"],
                linewidth=1.5, zorder=3)
        ax.fill_between(gens, k_hist, 1.0,
                         where=[k > 1 for k in k_hist],
                         color=self.C["orange"], alpha=0.15, zorder=2)
        ax.fill_between(gens, k_hist, 1.0,
                         where=[k < 1 for k in k_hist],
                         color=self.C["blue"], alpha=0.10, zorder=2)
        ax.axhline(y=1.0, color=self.C["green"],
                   linestyle="--", linewidth=1, alpha=0.6, zorder=4,
                   label="k=1 (критичность)")
        ax.axvline(x=self.current_gen+1,
                   color=self.C["yellow"], linewidth=1,
                   alpha=0.6, linestyle=":", zorder=5)

        ax.set_xlabel("Поколение", color=self.C["dim"], fontsize=7)
        ax.set_ylabel("k",         color=self.C["dim"], fontsize=7)
        ax.tick_params(colors=self.C["dim"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(self.C["border"])
        ax.legend(fontsize=7, facecolor=self.C["panel"],
                  edgecolor=self.C["border"],
                  labelcolor=self.C["text"], framealpha=0.8)

    # ── тепловая карта делений ────────────────────────────────────
    def _draw_heatmap(self, radius):
        ax = self.ax_heat
        ax.clear()
        ax.set_facecolor(self.C["bg"])
        ax.set_aspect("equal")
        ax.set_title("Карта делений (все поколения)",
                     color=self.C["dim"], fontsize=8)

        h = np.zeros((30, 30))
        edges = np.linspace(-radius, radius, 31)
        for frame in self.result["fiss_frames"]:
            if len(frame):
                values, _, _ = np.histogram2d(
                    frame[:, 0], frame[:, 1], bins=(edges, edges))
                h += values
        xedge = yedge = edges
        if h.any():
            h = np.ma.masked_where(h == 0, h)
            cmap = plt.cm.inferno
            cmap.set_bad(color=self.C["bg"])
            ax.pcolormesh(xedge, yedge, h.T, cmap=cmap, zorder=2)

        ax.add_patch(patches.Circle((0,0), radius,
                                     fill=False, linewidth=1,
                                     edgecolor=self.C["react_bd"],
                                     linestyle="--", zorder=3))
        ax.set_xlim(-radius*1.2, radius*1.2)
        ax.set_ylim(-radius*1.2, radius*1.2)
        ax.tick_params(colors=self.C["dim"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(self.C["border"])

    # ── гистограмма судеб ────────────────────────────────────────
    def _draw_bar(self):
        ax = self.ax_bar
        ax.clear()
        ax.set_facecolor(self.C["panel"])
        ax.set_title("Баланс нейтронов (текущее поколение)",
                     color=self.C["dim"], fontsize=8)

        if not self.result or not self.result["generation_stats"]:
            return
        gen = max(0, min(self.current_gen, len(self.result["generation_stats"])-1))
        counts = self.result["generation_stats"][gen]["outcome_counts"]
        labels = ["деление", "поглощение", "утечка", "не завершены"]
        values = [counts[key] for key in ("fission", "absorption", "leak", "unfinished")]
        colors = [self.C["orange"], self.C["purple"], self.C["blue"], self.C["dim"]]
        bars   = ax.bar(labels, values, color=colors, alpha=0.8,
                        edgecolor=self.C["border"], linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        str(val), ha="center", va="bottom",
                        color=self.C["text"], fontsize=8)

        ax.tick_params(colors=self.C["dim"], labelsize=8)
        ax.set_ylabel("нейтронов", color=self.C["dim"], fontsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(self.C["border"])

    # ── статус-панель ────────────────────────────────────────────
    def _update_status(self, msg, color):
        ax = self.ax_stat
        ax.clear()
        ax.set_facecolor(self.C["panel"])
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                color=color, fontsize=11, transform=ax.transAxes)

    def _update_status_full(self):
        r   = self.result
        ax  = self.ax_stat
        ax.clear()
        ax.set_facecolor(self.C["panel"])
        ax.axis("off")

        k    = r["k_mean"]
        kstd = r["k_std"]
        unfinished = sum(s["outcome_counts"]["unfinished"]
                         for s in r["generation_stats"])
        no_estimate = not r["k_history"]

        if no_estimate:
            status_text = "НЕТ ОЦЕНКИ"
            status_color = self.C["dim"]
            status_desc = "нет обработанных поколений"
        elif unfinished:
            status_text = "НЕПОЛНЫЙ РЕЗУЛЬТАТ"
            status_color = self.C["yellow"]
            status_desc = f"Лимит шагов: {unfinished} историй"
        elif k > 1.01:
            status_text  = "НАДКРИТИЧНЫЙ"
            status_color = self.C["orange"]
            status_desc  = "реакция разгоняется"
        elif k < 0.99:
            status_text  = "ПОДКРИТИЧНЫЙ"
            status_color = self.C["blue"]
            status_desc  = "реакция затухает"
        else:
            status_text  = "КРИТИЧНЫЙ"
            status_color = self.C["green"]
            status_desc  = "стационарный режим"

        y = 0.95
        def line(text, color=None, size=9, dy=0.07, bold=False):
            nonlocal y
            y -= dy * 0.8
            ax.text(0.5, y, text, ha="center", va="top",
                    color=color or self.C["dim"],
                    fontsize=size, transform=ax.transAxes,
                    fontweight="bold" if bold else "normal",
                    fontfamily="monospace")

        line("РЕЗУЛЬТАТ", self.C["dim"], size=8, dy=0.04)
        line("─────────────────", self.C["border"], size=8, dy=0.04)
        line("k: нет оценки" if no_estimate else f"k (источник) = {k:.4f}",
             self.C["text"], size=12, bold=True, dy=0.10)
        line(f"σ поколений: {kstd:.4f}", self.C["dim"], size=8, dy=0.07)
        line(status_text, status_color, size=11, bold=True, dy=0.09)
        line(status_desc, status_color, size=8, dy=0.06)
        line("─────────────────", self.C["border"], size=8, dy=0.08)

        mat_name = self.radio.value_selected
        mat      = self.result_config["material"] if self.result_config else MATERIALS[mat_name]
        radius = self.result_radius if self.result_radius is not None else self.sl_radius.val
        n_source = self.result_config["n_source"] if self.result_config else int(self.sl_n.val)
        line(f"Материал: {mat.name}", size=8, dy=0.06)
        line(f"Радиус:   {radius:.0f} см", size=8, dy=0.05)
        line(f"Нейтронов/п.: {n_source}", size=8, dy=0.05)
        line(f"Поколений:    {len(r['k_history'])}", size=8, dy=0.05)
        line("─────────────────", self.C["border"], size=8, dy=0.07)
        line(f"λ (св.пробег): {mat.mean_free_path():.3f} см", size=8, dy=0.06)
        line(f"ν (нейтр./дел): {mat.nu}", size=8, dy=0.05)

        # интерпретация
        line("─────────────────", self.C["border"], size=8, dy=0.08)
        if unfinished:
            hint = "Оценка может быть неполной.\nУвеличьте лимит шагов."
        elif no_estimate:
            hint = "Нет данных для классификации"
        elif k > 1.01:
            hint = "Уменьшите\nрадиус или обогащение"
        elif k < 0.99:
            hint = "Увеличьте\nрадиус или обогащение"
        else:
            hint = "Конфигурация\nоптимальна!"
        line(hint, self.C["yellow"], size=8, dy=0.09)

    # ── анимация по поколениям ────────────────────────────────────
    def _on_anim(self, event):
        if self.result is None or not self.result["k_history"]:
            return
        if self.anim_running:
            if self.anim_obj:
                self.anim_obj.event_source.stop()
            self.anim_running = False
            self.btn_anim.label.set_text("⟳  Анимация")
            return

        self.anim_running = True
        self.btn_anim.label.set_text("⏹  Стоп")
        radius = self.result_radius or float(self.sl_radius.val)
        n_gens = len(self.result["k_history"])

        def update(frame):
            self.current_gen = frame % n_gens
            self._draw_trajectories(radius)
            self._draw_k_chart()
            self._draw_bar()
            return []

        self.anim_obj = animation.FuncAnimation(
            self.fig, update, frames=n_gens,
            interval=600, repeat=True, blit=False)
        self.fig.canvas.draw_idle()

    def _step_gen(self, delta):
        if self.result is None or not self.result["k_history"]:
            return
        n = len(self.result["k_history"])
        self.current_gen = (self.current_gen + delta) % n
        self._draw_trajectories(self.result_radius or float(self.sl_radius.val))
        self._draw_k_chart()
        self._draw_bar()
        self.fig.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════
#  ЗАПУСК
# ══════════════════════════════════════════════════════════════════════

