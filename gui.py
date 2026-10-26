"""
БЛОК 4 — ИНТЕРФЕЙС (NeutronSimGUI).

Графический интерфейс симулятора на Matplotlib + блок запуска.
Соответствует БЛОКУ 4 и блоку «ЗАПУСК» исходного монолита NSPM MCNP 3000.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from collections import deque

from numba_compat import HAS_NUMBA
from material import MATERIALS
from physics import estimate_k_full


class NeutronSimGUI:
    def __init__(self):
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

        self.result       = None
        self.current_gen  = 0
        self.anim_running = False
        self.anim_obj     = None

        self._build_layout()
        self._build_controls()
        self._draw_empty_reactor()
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
        core = patches.Circle((0,0), r, color=self.C["reactor"],
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
    def _on_run(self, event):
        if self.anim_obj:
            self.anim_obj.event_source.stop()
            self.anim_obj = None

        mat_name = self.radio.value_selected
        mat      = MATERIALS[mat_name]
        radius   = float(self.sl_radius.val)
        n_source = int(self.sl_n.val)
        n_gen    = int(self.sl_gen.val)

        # ИСПРАВЛЕНИЕ: Принудительное обновление интерфейса, чтобы окно не зависало на время расчетов
        self._update_status("⏳  Симуляция...", self.C["yellow"])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.result = estimate_k_full(
            n_gen=n_gen, n_source=n_source,
            radius=radius, material=mat)

        self.current_gen = len(self.result["k_history"]) - 1
        self._render_all(radius)

    # ── рендер всех панелей ────────────────────────────────────────
    def _render_all(self, radius=None):
        if self.result is None:
            return
        if radius is None:
            radius = float(self.sl_radius.val)

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

        gen_idx = min(self.current_gen, len(self.result["path_frames"])-1)
        gen_idx = max(0, gen_idx)
        title = f"Поколение {gen_idx+1}/{len(self.result['k_history'])}  |  " \
                f"k = {self.result['k_history'][gen_idx]:.4f}"
        ax.set_title(title, color=self.C["text"], fontsize=9, pad=4)

        # замедлитель
        ax.add_patch(patches.Circle((0,0), radius*1.15,
                                    color="#0A2010", zorder=1))
        # активная зона
        ax.add_patch(patches.Circle((0,0), radius,
                                    color=self.C["reactor"],
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
            None:         self.C["dim"],
        }
        paths = self.result["path_frames"][gen_idx]
        for path, fate in paths:
            if len(path) < 2:
                continue
            pts = np.array(path)
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

        all_pts = []
        for frame in self.result["fiss_frames"]:
            all_pts.extend(frame)

        if all_pts:
            pts = np.array(all_pts)
            h, xedge, yedge = np.histogram2d(
                pts[:,0], pts[:,1], bins=30,
                range=[[-radius, radius], [-radius, radius]])
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

        gen = min(self.current_gen, len(self.result["path_frames"])-1)
        paths = self.result["path_frames"][gen]
        counts = {"деление": 0, "поглощение": 0, "утечка": 0}
        fate_map = {"fission": "деление",
                    "absorption": "поглощение",
                    "leak": "утечка"}
        for _, fate in paths:
            key = fate_map.get(fate)
            if key:
                counts[key] += 1

        labels = list(counts.keys())
        values = list(counts.values())
        colors = [self.C["orange"], self.C["purple"], self.C["blue"]]
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

        if k > 1.01:
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
            y -= dy
            ax.text(0.5, y, text, ha="center", va="top",
                    color=color or self.C["dim"],
                    fontsize=size, transform=ax.transAxes,
                    fontweight="bold" if bold else "normal",
                    fontfamily="monospace")

        line("РЕЗУЛЬТАТ", self.C["dim"], size=8, dy=0.04)
        line("─────────────────", self.C["border"], size=8, dy=0.04)
        line(f"k_eff = {k:.4f}", self.C["text"], size=14, bold=True, dy=0.10)
        line(f"± {kstd:.4f}", self.C["dim"], size=9, dy=0.07)
        line(status_text, status_color, size=11, bold=True, dy=0.09)
        line(status_desc, status_color, size=8, dy=0.06)
        line("─────────────────", self.C["border"], size=8, dy=0.08)

        mat_name = self.radio.value_selected
        mat      = MATERIALS[mat_name]
        line(f"Материал: {mat.name}", size=8, dy=0.06)
        line(f"Радиус:   {self.sl_radius.val:.0f} см", size=8, dy=0.05)
        line(f"Нейтронов/п.: {int(self.sl_n.val)}", size=8, dy=0.05)
        line(f"Поколений:    {len(r['k_history'])}", size=8, dy=0.05)
        line("─────────────────", self.C["border"], size=8, dy=0.07)
        line(f"λ (св.пробег): {mat.mean_free_path():.3f} см", size=8, dy=0.06)
        line(f"ν (нейтр./дел): {mat.nu}", size=8, dy=0.05)

        # интерпретация
        line("─────────────────", self.C["border"], size=8, dy=0.08)
        if k > 1.01:
            hint = "Уменьшите\nрадиус или обогащение"
        elif k < 0.99:
            hint = "Увеличьте\nрадиус или обогащение"
        else:
            hint = "Конфигурация\nоптимальна!"
        line(hint, self.C["yellow"], size=8, dy=0.09)

    # ── анимация по поколениям ────────────────────────────────────
    def _on_anim(self, event):
        if self.result is None:
            return
        if self.anim_running:
            if self.anim_obj:
                self.anim_obj.event_source.stop()
            self.anim_running = False
            self.btn_anim.label.set_text("⟳  Анимация")
            return

        self.anim_running = True
        self.btn_anim.label.set_text("⏹  Стоп")
        radius = float(self.sl_radius.val)
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
        if self.result is None:
            return
        n = len(self.result["k_history"])
        self.current_gen = (self.current_gen + delta) % n
        self._draw_trajectories(float(self.sl_radius.val))
        self._draw_k_chart()
        self._draw_bar()
        self.fig.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════
#  ЗАПУСК
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("NeutronMC — запуск интерфейса...")
    print("Зависимости: numpy, matplotlib")
    print("Опционально: numba (для ускорения)")
    print()
    if HAS_NUMBA:
        print("✓ Numba обнаружена — используется JIT-компиляция для физики")
    else:
        print("⚠ Numba не найдена — используется Python (медленнее)")
        print("  Установка: pip install numba")
    print()
    print("Режим: массивная генерация нейтронов с NumPy")
    print()
    app = NeutronSimGUI()





