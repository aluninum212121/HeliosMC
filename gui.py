"""Experiment workspace using standard ttk and embedded Matplotlib.

Agg uses the same controller and renderers without creating a Tk window.
Physics, statistics and research serialization stay in their existing modules.
"""
from copy import deepcopy
from pathlib import Path
from queue import Queue, Empty, Full
from types import SimpleNamespace
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
import numpy as np

from material import (MATERIALS, MaterialDataset, MaterialQuantity, DataOrigin,
                      get_reference_profiles, get_reference_transport_dataset)
from physics import (estimate_k_full, DEFAULT_MAX_MEMORY_BYTES,
                     create_slab_configuration, run_slab_experiment,
                     export_slab_experiment, load_slab_experiment)
from simulation_worker import SimulationWorker, ExecutionState


class NeutronSimGUI:
    SCENARIOS = ("Поколения · legacy", "Транспорт через слой")
    DEFAULTS = dict(scenario=SCENARIOS[0], material_mode="Reference",
        material=next(iter(MATERIALS)), profile="jendl5_point_0p0253ev",
        radius="25", source="150", generations="30", width="1", histories="10000",
        absorption="0.5", scattering="0", density="", density_sample="",
        density_method="", seed="", max_steps="500", max_collisions="1000", samples="40")
    TAB_NAMES = dict(overview="Обзор", heatmap="Тепловая карта", graphs="Графики",
                     trajectories="Траектории", data="Данные")
    COLORS = dict(bg="#F6F8FB", panel="#FFFFFF", border="#E2E8F0", text="#1E293B",
        dim="#64748B", blue="#2563EB", green="#16856A", yellow="#B7791F",
        orange="#C2413B", purple="#8170A6")
    OUTCOME_LABELS = dict(fission="Деление", absorption="Поглощение", leak="Утечка",
        unfinished="Не завершены", absorbed="Поглощены", escaped_left="Вышли слева",
        escaped_right="Прошли слой", collision_limit="Лимит столкновений", numeric_limit="Численный лимит")
    FIELD_LABELS = dict(radius="Радиус", source="Начальный источник", generations="Число поколений",
        width="Толщина слоя", histories="Число историй", absorption="Коэффициент поглощения",
        scattering="Коэффициент рассеяния", density="Плотность", seed="Seed",
        max_steps="Лимит шагов", max_collisions="Лимит столкновений", samples="Число сохраняемых траекторий")

    def __init__(self, show=True, *, native=None):
        self.native = str(matplotlib.get_backend()).lower() == "tkagg" if native is None else native
        self.C = dict(self.COLORS)
        self.worker = SimulationWorker()
        self._closed = False
        self.inputs = dict(self.DEFAULTS)
        self.variables, self.choice_maps, self.widgets = {}, {}, []
        self.root = None
        self.btn_run = SimpleNamespace(active=True)
        self.seed = None
        self.max_samples_per_gen, self.max_steps = 40, 500
        self.max_memory_bytes = DEFAULT_MAX_MEMORY_BYTES
        self.result = self.result_config = self.experiment = self.loaded_configuration = None
        self.result_radius = None
        self.result_scenario, self.current_gen = "generations", 0
        self.anim_running, self.anim_obj = False, None
        self.heat_bins, self.balance_fractions = 40, False
        self.balance_scope = "Текущее поколение"
        self.graph_kind, self.active_tab = "Коэффициент поколений", "overview"
        self._colorbar = self._heat_cache = None
        self._dirty_tabs = set(self.TAB_NAMES)
        self._progress = Queue(maxsize=1)
        self._after_id = self._started_at = None
        self.status_text = "Готово к новому эксперименту"
        self.data_rows, self.warning_text, self.summary_text, self.cards = [], "", "", []
        self._build_figures()
        if self.native:
            self._build_window()
            self._on_scenario_change()
            self._after_id = self.root.after(100, self._poll_worker)
        self._timer = self.fig.canvas.new_timer(interval=100)
        self._draw_empty()
        if show:
            self.root.mainloop() if self.native else plt.show()

    def _get(self, name):
        if name in self.variables:
            value = self.variables[name].get()
            return self.choice_maps.get(name, {}).get(value, value)
        return self.inputs[name]

    def _set(self, name, value):
        self.inputs[name] = str(value)
        if name in self.variables:
            inverse = {v: k for k, v in self.choice_maps.get(name, {}).items()}
            self.variables[name].set(inverse.get(str(value), str(value)))

    @property
    def execution_state(self):
        return self.worker.state

    @property
    def max_steps(self):
        return self._parse_int("max_steps")

    @max_steps.setter
    def max_steps(self, value):
        self._set("max_steps", value)

    @property
    def max_samples_per_gen(self):
        return self._parse_int("samples")

    @max_samples_per_gen.setter
    def max_samples_per_gen(self, value):
        self._set("samples", value)

    def _is_slab(self):
        return self._get("scenario") == self.SCENARIOS[1]

    def _build_figures(self):
        self.figures = {}
        for key in ("overview", "heatmap", "graphs", "trajectories"):
            figure = Figure(figsize=(8, 5), facecolor="white", layout="constrained")
            FigureCanvasAgg(figure)
            self.figures[key] = figure
        self.fig = self.figures["overview"]
        self.ax_overview = self.fig.add_subplot(111)
        # Status artists retain the previous headless integration surface.
        # Native metrics/status are ttk widgets, outside the scientific figure.
        self.ax_stat = Figure().add_subplot(111)
        self.ax_main = self.figures["trajectories"].add_subplot(111)
        self.ax_heat = self.figures["heatmap"].add_subplot(111)
        graph = self.figures["graphs"]
        self.ax_k = graph.add_subplot(111, label="coefficient")
        self.ax_bar = graph.add_subplot(111, label="balance")
        self.ax_population = graph.add_subplot(111, label="population")
        self.ax_bar.set_visible(False)
        self.ax_population.set_visible(False)

    def _build_window(self):
        import sys
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        if sys.platform == "win32":
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                pass  # Older Windows, or the host already chose DPI awareness.
        translations = {"Home": ("Сброс", "Исходный масштаб"), "Back": ("Назад", "Предыдущий вид"),
            "Forward": ("Вперёд", "Следующий вид"), "Pan": ("Перемещение", "Перемещение области просмотра"),
            "Zoom": ("Масштаб", "Выделить область для увеличения"), "Save": ("Сохранить", "Сохранить график в PNG")}
        controller = self
        class PlotToolbar(NavigationToolbar2Tk):
            # Keep Matplotlib's internal Pan/Zoom keys; translate visible labels.
            toolitems = tuple((name, translations.get(name, (name, tip))[1], icon, action)
                              for name, tip, icon, action in NavigationToolbar2Tk.toolitems if name != "Subplots")

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                for name, button in self._buttons.items():
                    button.configure(text=translations.get(name, (name, ""))[0])

            def save_figure(self, *args):
                controller._plot_dialog()
        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.root.title("HeliosMC — вычислительные эксперименты")
        w, h = min(1500, self.root.winfo_screenwidth()-60), min(900, self.root.winfo_screenheight()-80)
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(1000, 620)
        self.root.configure(background=self.C["bg"])
        for figure in self.figures.values():
            figure.set_dpi(self.root.winfo_fpixels("1i"))
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._on_close(None))
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=self.C["bg"], foreground=self.C["text"])
        style.configure("TFrame", background=self.C["bg"])
        style.configure("Panel.TFrame", background="white")
        style.configure("TLabel", background=self.C["bg"])
        style.configure("Panel.TLabel", background="white")
        style.configure("Muted.TLabel", foreground=self.C["dim"])
        style.configure("PanelMuted.TLabel", background="white", foreground=self.C["dim"], font=("Segoe UI", 9))
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabel", background="white", font=("Segoe UI", 11, "bold"))
        style.configure("Value.TLabel", background="white", font=("Segoe UI", 19, "bold"))
        style.configure("TButton", padding=(10, 7), background="#EEF2F7", borderwidth=0)
        style.map("TButton", background=[("active", "#E2E8F0")], foreground=[("disabled", "#94A3B8")])
        style.configure("Primary.TButton", background=self.C["blue"], foreground="white")
        style.map("Primary.TButton", background=[("disabled", "#B7C9EA"), ("active", "#1D4ED8")], foreground=[("disabled", "white")])
        style.configure("TEntry", fieldbackground="white", padding=5)
        style.configure("TCombobox", padding=5, fieldbackground="white")
        style.configure("TNotebook", background=self.C["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 9))
        style.map("TNotebook.Tab", background=[("selected", "white")], foreground=[("selected", self.C["blue"])])
        style.configure("Treeview", rowheight=30, background="white", fieldbackground="white", borderwidth=0)
        style.configure("Treeview.Heading", background="#EEF2F7", padding=8, font=("Segoe UI", 10, "bold"))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        top = ttk.Frame(self.root, padding=(18, 12))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        brand = ttk.Frame(top)
        brand.grid(row=0, column=0, sticky="w")
        ttk.Label(brand, text="HeliosMC", style="Title.TLabel").pack(anchor="w")
        ttk.Label(brand, text="Образовательный симулятор транспорта нейтронов",
                  style="Muted.TLabel", font=("Segoe UI", 9)).pack(anchor="w")
        actions = ttk.Frame(top)
        actions.grid(row=0, column=1, sticky="e")
        self.action_buttons = {}
        for key, title, command in (("new", "Новый", self.new_experiment), ("open", "Открыть", self._open_dialog)):
            button = ttk.Button(actions, text=title, command=command)
            button.pack(side="left", padx=3)
            self.action_buttons[key] = button
        self.export_menu = tk.Menu(self.root, tearoff=False)
        self.export_menu.add_command(label="Эксперимент: JSON / CSV / NPZ", command=self._export_dialog, state="disabled")
        self.export_menu.add_command(label="Текущий график: PNG, 300 dpi", command=self._plot_dialog, state="disabled")
        ttk.Menubutton(actions, text="Экспорт", menu=self.export_menu).pack(side="left", padx=6)
        for key, title, command, appearance in (("run", "Запустить", self._on_run, "Primary.TButton"), ("cancel", "Остановить", self._on_cancel, "TButton")):
            button = ttk.Button(actions, text=title, command=command, style=appearance)
            button.pack(side="left", padx=3)
            self.action_buttons[key] = button
        self.action_buttons["cancel"].state(["disabled"])
        body = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body, style="Panel.TFrame", width=300)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        sidebar.grid_propagate(False)
        sidebar.rowconfigure(0, weight=1)
        sidebar.columnconfigure(0, weight=1)
        self.sidebar_canvas = tk.Canvas(sidebar, width=280, bg="white", highlightthickness=0)
        self.sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(sidebar, orient="vertical", command=self.sidebar_canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=scroll.set)
        self.form = ttk.Frame(self.sidebar_canvas, style="Panel.TFrame", padding=14)
        form_window = self.sidebar_canvas.create_window((0, 0), window=self.form, anchor="nw")
        self.form.bind("<Configure>", lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda e: self.sidebar_canvas.itemconfigure(form_window, width=e.width))
        self.root.bind("<MouseWheel>", self._scroll_sidebar, add="+")
        self._build_form()
        workspace = ttk.Frame(body)
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(1, weight=1)
        banner = ttk.Frame(workspace, padding=(12, 8))
        banner.grid(row=0, column=0, sticky="ew")
        banner.columnconfigure(0, weight=1)
        self.warning_var = tk.StringVar(value="Legacy: исторические параметры; физическая валидация показателя не выполнена.")
        self.warning_label = ttk.Label(banner, textvariable=self.warning_var, foreground=self.C["yellow"], wraplength=700)
        self.warning_label.grid(row=0, column=0, sticky="w")
        ttk.Button(banner, text="Подробнее", command=self._show_details).grid(row=0, column=1, padx=(8, 0))
        banner.bind("<Configure>", lambda e: self.warning_label.configure(wraplength=max(200, e.width-130)))
        self.notebook = ttk.Notebook(workspace)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.pages, self.canvases = {}, {}
        for key, name in self.TAB_NAMES.items():
            page = ttk.Frame(self.notebook, style="Panel.TFrame", padding=10)
            self.notebook.add(page, text=name)
            self.pages[key] = page
            if key == "data":
                self._build_data_table(page)
                continue
            if key == "overview":
                self._build_cards(page)
                self.summary_var = tk.StringVar(value="Выберите сценарий и настройте параметры слева.")
                self.summary_label = ttk.Label(page, textvariable=self.summary_var, style="PanelMuted.TLabel", wraplength=900)
                self.summary_label.pack(fill="x", padx=8, pady=(6, 4))
                page.bind("<Configure>", lambda e: self.summary_label.configure(wraplength=max(200, e.width-40)))
            elif key == "graphs":
                self._build_graph_controls(page)
            elif key == "heatmap":
                row = ttk.Frame(page, style="Panel.TFrame")
                row.pack(fill="x", padx=4, pady=(0, 6))
                ttk.Label(row, text="Ячеек на ось", style="Panel.TLabel").pack(side="left", padx=(0, 8))
                self.bin_var = tk.StringVar(value=str(self.heat_bins))
                bins = ttk.Combobox(row, textvariable=self.bin_var, values=(20, 40, 80, 120), width=6, state="readonly")
                bins.pack(side="left")
                bins.bind("<<ComboboxSelected>>", self._change_bins)
            elif key == "trajectories":
                self._build_trajectory_controls(page)
            canvas = FigureCanvasTkAgg(self.figures[key], master=page)
            toolbar = PlotToolbar(canvas, page, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.canvases[key] = canvas
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        footer = ttk.Frame(self.root, padding=(16, 7))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value=self.status_text)
        self.status_label = ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w")
        footer.bind("<Configure>", lambda e: self.status_label.configure(wraplength=max(300, e.width-220)))
        self.progressbar = ttk.Progressbar(footer, mode="determinate", length=180)
        self.progressbar.grid(row=0, column=1, sticky="e", padx=(12, 0))

    def _build_form(self):
        t = self.ttk
        def group(title):
            frame = t.Frame(self.form, style="Panel.TFrame")
            frame.pack(fill="x", pady=(0, 16))
            t.Label(frame, text=title, style="Section.TLabel").pack(anchor="w", pady=(0, 8))
            return frame
        t.Label(self.form, text="Параметры эксперимента", style="Section.TLabel").pack(anchor="w", pady=(0, 18))
        self._combo(group("Сценарий"), "scenario", self.SCENARIOS, self._on_scenario_change)
        self.legacy_group = group("Материал · legacy")
        self._combo(self.legacy_group, "material", list(MATERIALS), self._form_changed)
        t.Label(self.legacy_group, text="Исторические константы. Источники не подтверждены.",
                style="PanelMuted.TLabel", wraplength=245).pack(anchor="w", pady=(6, 0))
        self.slab_group = group("Среда слоя")
        self._combo(self.slab_group, "material_mode", {"Reference · справочные": "Reference", "Учебная среда": "Учебная среда"}, self._on_scenario_change)
        self.slab_material_frame = t.Frame(self.slab_group, style="Panel.TFrame")
        self._combo(self.slab_material_frame, "material", ["Вода (H₂O)", "Графит"], self._on_material_change)
        self.profile_frame = t.Frame(self.slab_group, style="Panel.TFrame")
        t.Label(self.profile_frame, text="Энергетический профиль", style="PanelMuted.TLabel").pack(anchor="w")
        self.profile_combo = self._combo(self.profile_frame, "profile", self._profile_choices(), self._profile_changed)
        self.profile_note = t.Label(self.profile_frame, style="PanelMuted.TLabel", wraplength=245)
        self.profile_note.pack(anchor="w", pady=(6, 0))
        self.density_frame = t.Frame(self.slab_group, style="Panel.TFrame")
        for name, label in (("density", "Плотность графита, г/см³"), ("density_sample", "Марка / образец / источник плотности"), ("density_method", "Метод определения плотности")):
            self._entry(self.density_frame, name, label)
        self.rates_frame = t.Frame(self.slab_group, style="Panel.TFrame")
        self._entry(self.rates_frame, "absorption", "Коэффициент поглощения, см⁻¹")
        self._entry(self.rates_frame, "scattering", "Коэффициент рассеяния, см⁻¹")
        t.Label(self.rates_frame, text="Заданные математические коэффициенты, без привязки к реальному веществу.", style="PanelMuted.TLabel", wraplength=245).pack(anchor="w", pady=4)
        self.geometry_group = group("Параметры расчёта")
        self.generation_fields = t.Frame(self.geometry_group, style="Panel.TFrame")
        for name, label in (("radius", "Радиус сферы, см"), ("source", "Начальных нейтронов"), ("generations", "Поколений")):
            self._entry(self.generation_fields, name, label)
        self.slab_fields = t.Frame(self.geometry_group, style="Panel.TFrame")
        self._entry(self.slab_fields, "width", "Толщина слоя, см")
        self._entry(self.slab_fields, "histories", "Число независимых историй")
        self.advanced_button = t.Button(self.form, text="▸ Дополнительные настройки", command=self._toggle_advanced)
        self.advanced_button.pack(fill="x", pady=(0, 8))
        self.advanced = t.Frame(self.form, style="Panel.TFrame")
        self._entry(self.advanced, "seed", "Seed (пусто — сформировать)")
        self.steps_frame = t.Frame(self.advanced, style="Panel.TFrame")
        self._entry(self.steps_frame, "max_steps", "Лимит шагов на историю")
        self.collisions_frame = t.Frame(self.advanced, style="Panel.TFrame")
        self._entry(self.collisions_frame, "max_collisions", "Лимит столкновений на историю")
        self.samples_frame = t.Frame(self.advanced, style="Panel.TFrame")
        self._entry(self.samples_frame, "samples", "Сохраняемых траекторий на поколение")
        t.Label(self.advanced, text="Seed фиксируется перед запуском. Совпадение результатов требует той же среды и вычислительного режима.", style="PanelMuted.TLabel", wraplength=245).pack(anchor="w", pady=8)
        self.archive_frame = t.Frame(self.form, style="Panel.TFrame")
        t.Label(self.archive_frame, text="Архив · только чтение", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
        self.archive_label = t.Label(self.archive_frame, style="PanelMuted.TLabel", wraplength=245)
        self.archive_label.pack(fill="x")

    def _entry(self, parent, name, label):
        self.ttk.Label(parent, text=label, style="PanelMuted.TLabel", wraplength=245).pack(anchor="w", pady=(5, 3))
        variable = self.tk.StringVar(value=self.inputs[name])
        self.variables[name] = variable
        entry = self.ttk.Entry(parent, textvariable=variable)
        entry.pack(fill="x", pady=(0, 3))
        entry.bind("<KeyRelease>", self._form_changed)
        self.widgets.append((entry, "normal"))
        return entry

    def _combo(self, parent, name, choices, command):
        values, current = list(choices), self.inputs[name]
        if isinstance(choices, dict):
            self.choice_maps[name] = choices
            current = {v: k for k, v in choices.items()}.get(current, current)
        variable = self.variables.get(name)
        if variable is None:
            variable = self.tk.StringVar(value=current)
            self.variables[name] = variable
        widget = self.ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        widget.pack(fill="x")
        widget.bind("<<ComboboxSelected>>", command)
        self.widgets.append((widget, "readonly"))
        return widget

    def _profile_choices(self, material=None):
        return {f"{p['source_library']} · " + ("спектр Максвелла" if p['spectrum_kind'] == 'flux_average' else "точка 0,0253 эВ"): key
            for key, p in get_reference_profiles().items() if material is None or material in p['available_materials']}

    def _toggle_advanced(self):
        if self.advanced.winfo_manager():
            self.advanced.pack_forget()
            self.advanced_button.configure(text="▸ Дополнительные настройки")
        else:
            self.advanced.pack(fill="x")
            self.advanced_button.configure(text="▾ Дополнительные настройки")

    def _scroll_sidebar(self, event):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget == self.form:
                self.sidebar_canvas.yview_scroll(int(-event.delta / 120), "units")
                break
            widget = getattr(widget, "master", None)

    def _on_scenario_change(self, event=None):
        if self.native:
            if self.loaded_configuration is not None:
                for frame in (self.legacy_group, self.slab_group, self.geometry_group, self.advanced_button, self.advanced):
                    frame.pack_forget()
                self.archive_frame.pack(fill="x", pady=(8, 16))
                config = self.loaded_configuration
                p = config["parameters"]
                lines = [f"Слоёв: {len(config['layers'])}", f"Границы, см: {p['edges_cm']}",
                         f"Историй: {p['n_histories']}", f"Источник: x={p['source_x']:g} см; μ={p['source_mu']:g}",
                         f"Лимит столкновений: {p['max_collisions']}", f"Seed: {p['seed']}"]
                for i, layer in enumerate(config["layers"]):
                    lines += [f"\nСлой {i+1}: {layer['name']}", f"Данные: {layer['data_kind']}",
                              f"Температура: {layer.get('temperature')} K"]
                lines += ["\nПовтор использует полную исходную конфигурацию. Все параметры и источники — во вкладке «Данные». Для редактирования выберите «Новый»."]
                self.archive_label.configure(text="\n".join(lines))
                return
            self.archive_frame.pack_forget()
            self.geometry_group.pack(fill="x", pady=(0, 16))
            self.advanced_button.pack(fill="x", pady=(0, 8))
            slab = self._is_slab()
            for frame in (self.legacy_group, self.slab_group):
                frame.pack_forget()
            (self.slab_group if slab else self.legacy_group).pack(fill="x", pady=(0, 16), before=self.geometry_group)
            for frame in (self.generation_fields, self.slab_fields, self.steps_frame, self.collisions_frame, self.samples_frame):
                frame.pack_forget()
            (self.slab_fields if slab else self.generation_fields).pack(fill="x")
            (self.collisions_frame if slab else self.steps_frame).pack(fill="x")
            if not slab:
                self.samples_frame.pack(fill="x")
            if slab and self._get("material_mode") == "Reference":
                if self._get("material") not in ("Вода (H₂O)", "Графит"):
                    self._set("material", "Вода (H₂O)")
            elif not slab and self._get("material") not in MATERIALS:
                self._set("material", next(iter(MATERIALS)))
            if slab:
                self._on_material_change()
        self._form_changed()

    def _on_material_change(self, event=None):
        if self.native:
            for frame in (self.slab_material_frame, self.profile_frame, self.density_frame, self.rates_frame):
                frame.pack_forget()
            if self._get("material_mode") == "Reference":
                self.slab_material_frame.pack(fill="x", pady=(8, 0))
                self.profile_frame.pack(fill="x", pady=(8, 0))
                if self._get("material") == "Графит":
                    self.density_frame.pack(fill="x", pady=(8, 0))
                choices, selected = self._profile_choices(self._get("material")), self._get("profile")
                self.choice_maps["profile"] = choices
                self.profile_combo.configure(values=list(choices))
                if selected not in choices.values():
                    selected = next(iter(choices.values()))
                self._set("profile", selected)
            else:
                self.rates_frame.pack(fill="x", pady=(8, 0))
            self._profile_changed()
        self._form_changed()

    def _profile_changed(self, event=None):
        if self.native:
            profile = get_reference_profiles().get(self._get("profile"))
            if profile:
                low, high = profile["energy_bounds_ev"]
                energy = f"Точка: {low:g} эВ; без усреднения." if low == high else f"Усреднение по потоку: {low:g}–{high:g} эВ; kT = 0,0253 эВ."
                self.profile_note.configure(text=energy + "\nМатериал: 300 K. Свободные атомы, изотропное рассеяние. Проверены числа источника; модель не валидирована.")
        self._form_changed()

    def _form_changed(self, event=None):
        if self.result is not None and self.worker.state != ExecutionState.RUNNING:
            self._set_status("Настройки изменены. Показаны результаты последнего завершённого запуска.")
        elif self.result is None and self.native:
            self.warning_text = self._selected_warning()
            self.warning_var.set(self.warning_text)

    def _selected_warning(self):
        if not self._is_slab():
            return "Legacy: исторические параметры; физическая валидация показателя не выполнена."
        if self._get("material_mode") != "Reference":
            return "Учебная среда: коэффициенты задаются пользователем и не являются измеренными свойствами вещества."
        text = "Reference: документированные ядерные данные; свободные атомы, без связанных состояний и изменения энергии."
        if self._get("material") == "Графит":
            text += " Плотность графита требует сведений об образце и остаётся неподтверждённым входом."
        return text

    def _build_cards(self, parent):
        row = self.ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=(6, 10))
        self.card_vars = []
        for i in range(4):
            row.columnconfigure(i, weight=1, uniform="cards")
            frame = self.ttk.Frame(row, style="Panel.TFrame", padding=(10, 6))
            frame.grid(row=0, column=i, sticky="nsew", padx=2)
            title, value, note = (self.tk.StringVar(value=text) for text in ("Результат", "—", "После расчёта"))
            self.ttk.Label(frame, textvariable=title, style="PanelMuted.TLabel", wraplength=180).pack(anchor="w")
            self.ttk.Label(frame, textvariable=value, style="Value.TLabel").pack(anchor="w", pady=(5, 2))
            self.ttk.Label(frame, textvariable=note, style="PanelMuted.TLabel", wraplength=180).pack(anchor="w")
            frame.bind("<Configure>", lambda e, f=frame: [child.configure(wraplength=max(70, e.width-20)) for child in (f.winfo_children()[0], f.winfo_children()[2])])
            self.card_vars.append((title, value, note))

    def _build_graph_controls(self, parent):
        row = self.ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=(0, 8))
        self.graph_var = self.tk.StringVar(value=self.graph_kind)
        self.graph_combo = self.ttk.Combobox(row, textvariable=self.graph_var, state="readonly", width=27,
            values=("Коэффициент поколений", "Баланс исходов", "Частицы по поколениям"))
        self.graph_combo.pack(side="left", padx=(0, 10))
        self.graph_combo.bind("<<ComboboxSelected>>", self._graph_changed)
        self.scope_var = self.tk.StringVar(value=self.balance_scope)
        self.scope_combo = self.ttk.Combobox(row, textvariable=self.scope_var, state="readonly", width=19,
            values=("Текущее поколение", "Весь эксперимент"))
        self.scope_combo.pack(side="left", padx=(0, 10))
        self.scope_combo.bind("<<ComboboxSelected>>", self._graph_changed)
        self.fractions_var = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(row, text="Доли", variable=self.fractions_var, command=self._graph_changed).pack(side="left")

    def _build_trajectory_controls(self, parent):
        row = self.ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=(0, 8))
        self.ttk.Button(row, text="Назад", command=lambda: self._step_gen(-1)).pack(side="left", padx=(0, 6))
        self.generation_var = self.tk.StringVar(value="—")
        self.generation_combo = self.ttk.Combobox(row, textvariable=self.generation_var, width=6, state="readonly")
        self.generation_combo.pack(side="left", padx=(0, 6))
        self.generation_combo.bind("<<ComboboxSelected>>", self._select_generation)
        self.ttk.Button(row, text="Вперёд", command=lambda: self._step_gen(1)).pack(side="left", padx=(0, 12))
        self.play_button = self.ttk.Button(row, text="Воспроизвести", command=self._on_anim)
        self.play_button.pack(side="left")
        self.ttk.Button(row, text="К первому", command=self._reset_generation).pack(side="left", padx=6)

    def _build_data_table(self, parent):
        self.ttk.Label(parent, text="Параметры последнего расчёта, результаты и происхождение данных", style="Section.TLabel").pack(anchor="w", pady=(4, 12))
        container = self.ttk.Frame(parent, style="Panel.TFrame")
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        columns = ("parameter", "value", "unit", "uncertainty", "source")
        self.data_tree = self.ttk.Treeview(container, columns=columns, show="headings")
        for key, title, width in zip(columns, ("Параметр", "Значение", "Единица", "Неопределённость", "Источник / статус"), (220, 170, 80, 220, 350)):
            self.data_tree.heading(key, text=title)
            self.data_tree.column(key, width=width, minwidth=70, stretch=key in ("parameter", "source"))
        self.data_tree.grid(row=0, column=0, sticky="nsew")
        vertical = self.ttk.Scrollbar(container, orient="vertical", command=self.data_tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = self.ttk.Scrollbar(container, orient="horizontal", command=self.data_tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.data_tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)

    def _parse_int(self, name, minimum=0):
        try:
            value = int(str(self._get(name)).strip())
            if value < minimum:
                raise ValueError()
            return value
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"Поле «{self.FIELD_LABELS.get(name, name)}»: требуется целое число не меньше {minimum}.") from None

    def _parse_float(self, name, positive=False):
        try:
            value = float(str(self._get(name)).strip().replace(",", "."))
            if not np.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError()
            return value
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"Поле «{self.FIELD_LABELS.get(name, name)}»: требуется конечное {'положительное' if positive else 'неотрицательное'} число.") from None

    def _seed_value(self):
        if str(self._get("seed")).strip():
            return self._parse_int("seed")
        value = self.seed if self.seed is not None else int(np.random.SeedSequence().entropy)
        self._set("seed", value)
        return value

    def _build_slab_configuration(self):
        reference = self._get("material_mode") == "Reference"
        if reference:
            profile_id = self._get("profile")
            profiles = get_reference_profiles()
            if profile_id not in profiles:
                raise ValueError("Неизвестный энергетический профиль.")
            profile = profiles[profile_id]
            material = self._get("material")
            if material not in profile["available_materials"]:
                raise ValueError("Этот reference-профиль недоступен для выбранного материала.")
            options = {}
            if material == "Графит":
                sample, method = str(self._get("density_sample")).strip(), str(self._get("density_method")).strip()
                if not sample or not method:
                    raise ValueError("Для графита укажите плотность, марку/образец и метод определения плотности.")
                options = dict(density=self._parse_float("density", True), density_origin=DataOrigin(
                    source_library="Пользовательские сведения о графите", original_reference=sample, processing_description=method))
            medium = get_reference_transport_dataset(material, approximation=profile["approximation"], profile_id=profile_id, **options)
        elif self._get("material_mode") == "Учебная среда":
            medium = MaterialDataset("gui:educational-medium", "Учебная среда", "educational",
                "Коэффициенты заданы пользователем", "1", ("Математическая среда; не измеренные свойства вещества.",),
                (MaterialQuantity("absorption_rate", "1/cm", self._parse_float("absorption")),
                 MaterialQuantity("scattering_rate", "1/cm", self._parse_float("scattering"))))
        else:
            raise ValueError("Неизвестный режим материала слоя.")
        return create_slab_configuration("gui-slab", [medium], [0., self._parse_float("width", True)],
            n_histories=self._parse_int("histories", 1), seed=self._seed_value(),
            max_collisions=self._parse_int("max_collisions"), max_samples=0,
            material_mode="reference" if reference else "legacy")

    def _set_running_controls(self, running):
        self.btn_run.active = not running
        if self.native:
            locked = running or self.loaded_configuration is not None
            for widget, normal in self.widgets:
                widget.configure(state="disabled" if locked else normal)
            for key in ("new", "open", "run"):
                self.action_buttons[key].state(["disabled"] if running else ["!disabled"])
            self.action_buttons["cancel"].state(["!disabled"] if running else ["disabled"])
            if not running:
                self.progressbar.stop()

    def _on_run(self, event=None):
        if self.worker.state == ExecutionState.RUNNING or self._closed:
            return
        self._stop_animation()
        try:
            if self._is_slab():
                config = deepcopy(self.loaded_configuration) if self.loaded_configuration is not None else self._build_slab_configuration()
                scenario, function, arguments = "slab", run_slab_experiment, dict(config=config)
            else:
                key = self._get("material")
                if key not in MATERIALS:
                    raise ValueError("Выберите существующий legacy-материал.")
                config = dict(n_gen=self._parse_int("generations", 1), n_source=self._parse_int("source", 1),
                    radius=self._parse_float("radius", True), material=MATERIALS[key], seed=self._seed_value(),
                    max_samples_per_gen=self._parse_int("samples"), max_steps=self._parse_int("max_steps"),
                    max_memory_bytes=self.max_memory_bytes)
                scenario, function, arguments = "generations", estimate_k_full, dict(config)
                arguments["callback"] = self._report_progress
            self.result_scenario, self.result_config = scenario, config
            self.result_radius = config.get("radius")
            self.result = self.experiment = None
            self._heat_cache = None
            self._drain_progress()
            self._started_at = time.perf_counter()
            self.worker.start(function, **arguments)
        except Exception as exc:
            self.worker.state = ExecutionState.FAILED
            self.result = self.experiment = None
            self._set_running_controls(False)
            self._update_status(f"Ошибка ввода / запуска: {exc}", self.C["orange"])
            if self.native:
                self.export_menu.entryconfigure(0, state="disabled")
                self.export_menu.entryconfigure(1, state="disabled")
        else:
            self._set_running_controls(True)
            self._draw_empty("Расчёт выполняется…")
            self.warning_text = self._selected_warning()
            if self.native:
                self.warning_var.set(self.warning_text)
            self._set_status("Выполняется расчёт…")
            if self.native:
                self.export_menu.entryconfigure(0, state="disabled")
                self.export_menu.entryconfigure(1, state="disabled")
                if scenario == "slab":
                    self.progressbar.configure(mode="indeterminate")
                    self.progressbar.start(15)
                else:
                    self.progressbar.configure(mode="determinate", maximum=config["n_gen"], value=0)

    def _report_progress(self, generation, k, fissions, absorptions, leakages):
        # At most one pending update; no widget access from the worker thread.
        try:
            self._progress.put_nowait((generation, k))
        except Full:
            try:
                self._progress.get_nowait()
            except Empty:
                pass
            self._progress.put_nowait((generation, k))

    def _drain_progress(self):
        latest = None
        while True:
            try:
                latest = self._progress.get_nowait()
            except Empty:
                return latest

    def _on_cancel(self):
        self.worker.cancel()
        self._set_status("Остановка запрошена; ожидание контрольной точки расчёта…")

    def _poll_worker(self):
        if self._closed:
            return
        if self.native:
            self._after_id = self.root.after(100, self._poll_worker)
        progress = self._drain_progress()
        if progress and self.worker.state == ExecutionState.RUNNING:
            self._set_status(f"Выполняется поколение {progress[0]} / {self.result_config['n_gen']}")
            if self.native:
                self.progressbar.configure(value=progress[0])
        message = self.worker.poll()
        if message is None:
            return
        state, payload = message
        self._set_running_controls(False)
        if state == ExecutionState.COMPLETED:
            try:
                if self.result_scenario == "slab":
                    self.experiment, self.result = payload, payload["result"]
                else:
                    self.result = payload
                self.current_gen = max(0, len(self.result.get("k_history", []))-1)
                self._heat_cache = None
                self._dirty_tabs = set(self.TAB_NAMES)
                self._render_all(self.result_radius)
                elapsed = time.perf_counter()-self._started_at if self._started_at else 0
                self._set_status(f"Расчёт завершён за {elapsed:.2f} с · Seed: {self.result.get('run_info', {}).get('seed', '—')}")
                if self.native:
                    self.notebook.select(self.pages["overview"])
                    self.export_menu.entryconfigure(0, state="normal" if self.experiment is not None else "disabled")
                    self.export_menu.entryconfigure(1, state="normal")
            except Exception as exc:
                self.worker.state = ExecutionState.FAILED
                self.result = self.experiment = None
                self._update_status(f"Ошибка отображения: {exc}", self.C["orange"])
        else:
            self.result = self.experiment = None
            self._update_status(f"Ошибка расчёта: {payload}" if state == ExecutionState.FAILED else "Расчёт остановлен. Итоговые результаты не сформированы.", self.C["orange"])

    def _style_axis(self, ax, title="", xlabel="", ylabel="", grid=True):
        ax.clear()
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=12, color=self.C["text"], loc="left", pad=16)
        ax.set_xlabel(xlabel, fontsize=10, color=self.C["dim"])
        ax.set_ylabel(ylabel, fontsize=10, color=self.C["dim"])
        ax.tick_params(colors=self.C["dim"], labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color(self.C["border"])
        ax.set_axisbelow(True)
        if grid:
            ax.grid(True, color="#EDF0F4", linewidth=.8)

    def _placeholder(self, ax, text):
        self._style_axis(ax, grid=False)
        ax.axis("off")
        ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes,
                color=self.C["dim"], fontsize=12, linespacing=1.7, wrap=True)

    def _draw_empty(self, message="Запустите эксперимент для отображения результатов"):
        self._remove_colorbar()
        for ax in (self.ax_overview, self.ax_k, self.ax_heat, self.ax_main, self.ax_bar, self.ax_population):
            self._placeholder(ax, message)
        self.cards = [("Результат", "—", "После расчёта")]*4
        self._sync_cards()
        if self.native:
            self.summary_var.set("Выберите сценарий и настройте параметры слева.")
            self.data_tree.delete(*self.data_tree.get_children())
        for figure in self.figures.values():
            figure.canvas.draw_idle()

    def _render_all(self, radius=None):
        if self.result is None:
            return
        if self.result_scenario == "generations":
            self.current_gen = max(0, min(self.current_gen, len(self.result["k_history"])-1))
        self._update_status_full()
        self._update_data_rows()
        self._dirty_tabs = set(self.TAB_NAMES)
        graph_choices = ("Баланс исходов",) if self.result_scenario == "slab" else ("Коэффициент поколений", "Баланс исходов", "Частицы по поколениям")
        if self.graph_kind not in graph_choices:
            self.graph_kind = graph_choices[0]
        if self.native:
            choices = list(range(1, len(self.result.get("k_history", []))+1))
            self.generation_combo.configure(values=choices)
            self.generation_var.set(str(self.current_gen+1) if choices else "—")
            self.graph_combo.configure(values=graph_choices)
            self.graph_var.set(self.graph_kind)
            self.scope_combo.configure(state="disabled" if self.result_scenario == "slab" else "readonly")
            self._render_tab(self.active_tab)
        else:
            self._draw_overview()
            self._draw_k_chart()
            self._draw_bar()
            self._draw_population()
            self._draw_heatmap(radius)
            self._draw_trajectories(radius)
            self._show_graph_axis()

    def _render_tab(self, tab):
        if tab not in self._dirty_tabs or self.result is None:
            return
        if tab == "overview": self._draw_overview()
        elif tab == "heatmap": self._draw_heatmap()
        elif tab == "trajectories": self._draw_trajectories()
        elif tab == "graphs":
            if self.graph_kind == "Коэффициент поколений": self._draw_k_chart()
            elif self.graph_kind == "Баланс исходов": self._draw_bar()
            else: self._draw_population()
            self._show_graph_axis()
        self._dirty_tabs.discard(tab)
        if tab in self.figures:
            self.figures[tab].canvas.draw_idle()

    def _on_tab_changed(self, event=None):
        selected = self.notebook.select()
        self.active_tab = next(k for k, page in self.pages.items() if str(page) == selected)
        if self.active_tab != "trajectories":
            self._stop_animation()
        self._render_tab(self.active_tab)
        self.export_menu.entryconfigure(1, state="normal" if self.result is not None and self.active_tab != "data" else "disabled")

    def _draw_overview(self):
        if self.result_scenario == "slab":
            self._draw_balance(self.ax_overview, fractions=True, all_generations=True)
        else:
            self._plot_coefficients(self.ax_overview, compact=True)

    def _draw_k_chart(self):
        if self.result_scenario == "slab":
            self._placeholder(self.ax_k, "Для транспорта через слой коэффициент поколений не рассчитывается.")
        else:
            self._plot_coefficients(self.ax_k)

    def _plot_coefficients(self, ax, compact=False):
        self._style_axis(ax, "Коэффициент воспроизводства источника (учебная модель)", "Поколение", "Отношение воспроизводства")
        history = self.result.get("k_history", [])
        if not history:
            self._placeholder(ax, "Нет обработанных поколений")
            return
        gens = np.arange(1, len(history)+1)
        ax.plot(gens, history, color=self.C["blue"], linewidth=1.7, label="k поколения")
        start = self.result["estimator"]["active_generation_start"]
        if start:
            ax.axvspan(.5, start+.5, color=self.C["dim"], alpha=.07)
        ax.axhline(self.result["k_mean"], color=self.C["purple"], linestyle="-.", linewidth=1, label=f"Среднее k = {self.result['k_mean']:.4f}")
        ax.axhline(1, color="#A3AEBE", linestyle=":", linewidth=1, label="Уровень 1")
        incomplete = [g for g, s in zip(gens, self.result["generation_stats"]) if s["outcome_counts"]["unfinished"]]
        if incomplete:
            ax.scatter(incomplete, [history[g-1] for g in incomplete], marker="x", color=self.C["yellow"], label="Лимит шагов", zorder=5)
        if not compact:
            ax.axvline(self.current_gen+1, color="#94A3B8", linewidth=.8, linestyle=":")
        ax.margins(x=.02, y=.12)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -.17), ncol=2, frameon=False, fontsize=9)

    def _draw_population(self):
        ax = self.ax_population
        self._style_axis(ax, "Размер источника по поколениям", "Поколение", "Обработано нейтронов")
        if self.result_scenario != "generations" or not self.result.get("generation_stats"):
            self._placeholder(ax, "В этом сценарии нет последовательности поколений.")
            return
        counts = [s["processed"] for s in self.result["generation_stats"]]
        ax.plot(range(1, len(counts)+1), counts, color=self.C["blue"], linewidth=1.7)
        ax.set_ylim(bottom=0)

    def _draw_bar(self):
        self._draw_balance(self.ax_bar, fractions=self.balance_fractions, all_generations=self.balance_scope == "Весь эксперимент")

    def _draw_balance(self, ax, *, fractions=False, all_generations=False):
        slab = self.result_scenario == "slab"
        if slab:
            counts, scope = self.result["outcome_counts"], "весь эксперимент"
        else:
            stats = self.result.get("generation_stats", [])
            if not stats:
                self._placeholder(ax, "Нет обработанных поколений")
                return
            if all_generations:
                counts = {key: sum(s["outcome_counts"][key] for s in stats) for key in stats[0]["outcome_counts"]}
                scope = "все поколения; повторные истории"
            else:
                gen = min(self.current_gen, len(stats)-1)
                counts, scope = stats[gen]["outcome_counts"], f"поколение {gen+1}"
        self._style_axis(ax, f"Баланс исходов · {scope}", "", "Доля обработанных историй" if fractions else "Число историй")
        keys = list(counts)
        if slab:
            keys = [k for k in keys if counts[k] or k in ("absorbed", "escaped_left", "escaped_right")]
        total = sum(counts.values())
        values = [counts[k]/total if fractions and total else counts[k] for k in keys]
        palette = [self.C["blue"], "#69A9A0", "#9CAEC6", "#B9A4C5", "#C9AC87"]
        bars = ax.bar([self.OUTCOME_LABELS[k] for k in keys], values, color=palette[:len(keys)], width=.55)
        for bar, key, value in zip(bars, keys, values):
            label = f"{value:.1%}\n(n={counts[key]})" if fractions else str(counts[key])
            ax.annotate(label, (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, color=self.C["text"])
        ax.set_ylim(0, max(values, default=0)*1.25 or 1)
        ax.grid(False, axis="x")

    def _remove_colorbar(self):
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

    def _draw_heatmap(self, radius=None):
        self._remove_colorbar()
        ax = self.ax_heat
        if self.result_scenario != "generations":
            self._placeholder(ax, "Пространственные события слоя не сохранены в исследовательском результате.\nДоступны агрегированные исходы во вкладках «Графики» и «Данные».")
            return
        radius = radius or self.result_radius or float(self._get("radius"))
        self._style_axis(ax, "Зарегистрированные деления · проекция X–Y, все поколения", "X, см", "Y, см", grid=False)
        bins = int(self.heat_bins)
        if not 10 <= bins <= 200:
            raise ValueError("Разрешение карты должно быть от 10 до 200 ячеек на ось.")
        cache_key = (id(self.result), bins, radius)
        if self._heat_cache is None or self._heat_cache[0] != cache_key:
            edges = np.linspace(-radius, radius, bins+1)
            hist = np.zeros((bins, bins))
            for points in self.result.get("fiss_frames", []):
                if len(points):
                    hist += np.histogram2d(points[:, 0], points[:, 1], bins=(edges, edges))[0]
            self._heat_cache = cache_key, hist, edges
        _, hist, edges = self._heat_cache
        if not hist.any():
            self._placeholder(ax, "В расчёте не зарегистрированы координаты делений.\nКарта нейтронного потока этим алгоритмом не рассчитывается.")
            return
        mesh = ax.pcolormesh(edges, edges, hist.T, cmap="viridis", shading="flat", rasterized=True)
        self._colorbar = self.figures["heatmap"].colorbar(mesh, ax=ax, pad=.035)
        self._colorbar.set_label("Зарегистрировано делений в ячейке", fontsize=10)
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="#64748B", linewidth=1.1, linestyle="--"))
        ax.set(xlim=(-radius*1.03, radius*1.03), ylim=(-radius*1.03, radius*1.03), aspect="equal")

    def _draw_trajectories(self, radius=None):
        ax = self.ax_main
        if self.result_scenario != "generations":
            self._placeholder(ax, "Траектории слоя не входят в исследовательский архив.\nПространственная информация не восстанавливается из агрегированных исходов.")
            return
        history = self.result.get("k_history", [])
        if not history:
            self._placeholder(ax, "Нет обработанных поколений")
            return
        radius = radius or self.result_radius or float(self._get("radius"))
        gen = max(0, min(self.current_gen, len(history)-1))
        title = f"Поколение {gen+1}/{len(history)}  |  k поколения = {history[gen]:.4f}"
        self._style_axis(ax, title, "X, см", "Y, см")
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="#64748B", linewidth=1.1))
        frames = self.result.get("path_frames", [])
        paths = frames[gen] if gen < len(frames) else []
        colors = dict(fission=self.C["orange"], absorption=self.C["purple"], leak=self.C["blue"], unfinished=self.C["dim"])
        seen = set()
        for path, fate in paths:
            points = np.asarray(path)
            if points.ndim != 2 or len(points) == 0:
                continue
            label = self.OUTCOME_LABELS.get(fate, fate) if fate not in seen else None
            seen.add(fate)
            ax.plot(points[:, 0], points[:, 1], color=colors.get(fate, self.C["dim"]), alpha=.65, linewidth=.9, label=label)
            ax.scatter(points[-1, 0], points[-1, 1], color=colors.get(fate, self.C["dim"]), s=13)
        ax.set_aspect("equal", adjustable="box")
        if not paths:
            ax.set_xlim(-radius*1.08, radius*1.08)
            ax.set_ylim(-radius*1.08, radius*1.08)
        else:
            ax.margins(.06)
        processed = self.result["generation_stats"][gen]["processed"]
        note = f"Выборка: {len(paths)} из {processed} историй. Проекция X–Y."
        if not paths:
            ax.text(.5, .5, "Траектории не сохранены для этого поколения", ha="center", transform=ax.transAxes, color=self.C["dim"])
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(0, -.17), ncol=2, frameon=False, fontsize=9)
        ax.set_xlabel("X, см\n"+note, fontsize=9)

    def _show_graph_axis(self):
        self.ax_k.set_visible(self.graph_kind == "Коэффициент поколений")
        self.ax_bar.set_visible(self.graph_kind == "Баланс исходов")
        self.ax_population.set_visible(self.graph_kind == "Частицы по поколениям")

    def _graph_changed(self, event=None):
        self.graph_kind, self.balance_scope = self.graph_var.get(), self.scope_var.get()
        self.balance_fractions = self.fractions_var.get()
        self._dirty_tabs.add("graphs")
        self._render_tab("graphs")

    def _change_bins(self, event=None):
        self.heat_bins = int(self.bin_var.get())
        self._dirty_tabs.add("heatmap")
        self._render_tab("heatmap")

    def _sync_cards(self):
        if self.native:
            for variables, values in zip(self.card_vars, self.cards):
                for variable, value in zip(variables, values):
                    variable.set(value)

    def _set_status(self, message):
        self.status_text = message
        if self.native:
            self.status_var.set(message)

    def _update_status(self, message, color=None):
        self._set_status(message)
        self.ax_stat.clear()
        self.ax_stat.text(.5, .5, message, ha="center", transform=self.ax_stat.transAxes)
        self._draw_empty(message)
        if self.native:
            self.warning_var.set(message)

    def _update_status_full(self):
        r = self.result
        self.ax_stat.clear()
        self.ax_stat.axis("off")
        if self.result_scenario == "slab":
            cards = []
            for outcome, title in (("escaped_right", "Прошли слой"), ("absorbed", "Поглощены"), ("escaped_left", "Вышли слева")):
                estimate = r["estimates"][outcome]
                value, se = estimate["estimate"], estimate["standard_error"]
                cards.append((title, "Недоступно" if value is None else f"{value:.2%}",
                    f"n = {r['outcome_counts'][outcome]}" if se is None else f"SE = {se:.4f}"))
            cards.append(("Завершено историй", str(r["completed_histories"]), f"из {r['processed_histories']}"))
            self.cards = cards
            config = self.result_config or (self.experiment or {}).get("configuration", {})
            mode = config.get("material_mode", "legacy")
            self.summary_text = f"Транспорт через слой · {'Reference' if mode == 'reference' else 'учебная среда'} · Seed: {r['run_info']['effective_seed']} · Толщина: {r['run_info']['edges_cm'][-1]:g} см"
            self.warning_text = ("Reference: проверены данные источника; транспортное приближение не валидировано." if mode == "reference" else
                "Учебная среда: коэффициенты заданы пользователем и не описывают измеренное вещество.")
            if r["unfinished_histories"]:
                self.warning_text += " Есть незавершённые истории; полные доли и интервалы недоступны."
            if mode == "reference" and any("graphite" in layer["material_id"] for layer in config.get("layers", [])):
                self.warning_text += " Плотность графита введена пользователем и не проверена."
            lines = [self.summary_text, self.warning_text]
        else:
            history, k, std = r["k_history"], r["k_mean"], r["k_std"]
            stats = r["generation_stats"]
            unfinished = sum(s["outcome_counts"]["unfinished"] for s in stats)
            processed = sum(s["processed"] for s in stats)
            info = r.get("run_info", {})
            initial = (self.result_config or {}).get("n_source", info.get("n_source", self._get("source")))
            self.cards = [("Среднее k", f"{k:.4f}" if history else "—", "Воспроизводство источника"),
                ("Разброс поколений σ", f"{std:.4f}" if history else "—", "Не ошибка среднего"),
                ("Завершено историй", str(processed-unfinished), f"из {processed}; за все поколения"),
                ("Поколений", str(len(history)), "Фактически обработано")]
            self.summary_text = f"Поколения · legacy · Начальный источник: {initial} · Seed: {info.get('seed', '—')}"
            material_info = info.get("material", {})
            if material_info:
                self.summary_text += f" · Материал: {material_info.get('name', '—')} · Радиус: {info.get('radius_cm', '—')} см"
            self.warning_text = "Legacy: исторические параметры; физическая валидация показателя не выполнена."
            lines = ["k: нет оценки" if not history else f"Среднее k = {k:.4f}"]
            if history:
                start = r["estimator"]["active_generation_start"]
                gen = max(0, min(self.current_gen, len(history)-1))
                lines += [f"По поколениям {start+1}–{len(history)}", f"σ поколений = {std:.4f}; не ошибка среднего",
                    f"k поколения {gen+1} = {history[gen]:.4f}", f"Нейтронов в поколении: {stats[gen]['processed']}"]
                self.summary_text += f" · Среднее по поколениям {start+1}–{len(history)}"
            if not history:
                lines.append("НЕТ ОЦЕНКИ")
            elif unfinished:
                lines += ["НЕПОЛНЫЙ РЕЗУЛЬТАТ", f"Лимит шагов: {unfinished}/{processed} ({unfinished/processed:.1%})", "Оценка может быть неполной"]
                self.warning_text += f" Лимит шагов: {unfinished}/{processed}; оценка может быть неполной."
            else:
                lines.append("СРЕДНЕЕ k > 1,01" if k > 1.01 else "СРЕДНЕЕ k < 0,99" if k < .99 else "СРЕДНЕЕ k ≈ 1")
            lines += [f"Начальный источник: {initial}", self.warning_text]
        for i, text in enumerate(lines):
            self.ax_stat.text(.02, .96-i*.85/max(1, len(lines)), text, transform=self.ax_stat.transAxes, fontsize=9)
        self._sync_cards()
        if self.native:
            self.summary_var.set(self.summary_text)
            self.warning_var.set(self.warning_text)

    def _update_data_rows(self):
        r, rows = self.result, []
        def add(name, value, unit="", uncertainty="—", source=""):
            if value is None: value = "Неизвестно / недоступно"
            if isinstance(value, float): value = f"{value:.8g}"
            rows.append((name, str(value), unit, uncertainty, source))
        if self.result_scenario == "slab":
            config = self.result_config or self.experiment["configuration"]
            add("Сценарий", "Транспорт через слой")
            add("Режим материала", config.get("material_mode", "legacy"))
            add("Версия алгоритма", r["algorithm_version"])
            add("Версия схемы", config["schema_version"])
            add("Идентификатор конфигурации", config["configuration_id"])
            add("Seed", config["parameters"]["seed"])
            for name, value in config["parameters"].items():
                if name != "seed": add(name, value, "см" if name in ("edges_cm", "source_x") else "", source="Сохранённая конфигурация")
            for name, count in r["outcome_counts"].items():
                estimate = r["estimates"].get(name)
                add(self.OUTCOME_LABELS[name], count, "историй")
                if estimate:
                    ci = estimate["confidence_interval"]
                    uncertainty = "Недоступна" if ci is None else f"SE={estimate['standard_error']:.6g}; {config['parameters']['confidence']:.0%} ДИ [{ci[0]:.6g}, {ci[1]:.6g}]"
                    add("Доля: "+self.OUTCOME_LABELS[name], estimate["estimate"], "1", uncertainty, "Монте-Карло; только статистическая неопределённость")
            profile = config.get("reference_profile")
            if profile:
                for name in ("source_library", "source_version", "energy_bounds_ev", "spectrum", "material_temperature_K", "averaging_method", "numerical_verification_status", "scientific_validation_status"):
                    add(name, profile[name], "эВ" if name == "energy_bounds_ev" else "K" if name == "material_temperature_K" else "", source="; ".join(profile["source_urls"]))
            for i, dataset in enumerate(config["layers"]):
                add(f"Слой {i+1}: материал", dataset["name"], source=dataset["source"])
                add(f"Слой {i+1}: температура", dataset.get("temperature"), "K")
                quality = dataset.get("quality") or {}
                add(f"Слой {i+1}: проверка источников", quality.get("verification_status"), source="Статус всего материала, включая плотность")
                state = dataset.get("state") or {}
                add(f"Слой {i+1}: плотность", state.get("density"), state.get("density_unit", "г/см³"), source=str(state.get("origin") or "Источник не указан"))
                for quantity in dataset["quantities"]:
                    origin = quantity.get("origin") or {}
                    add(quantity["name"], quantity["values"], quantity["unit"] or "", "Не задана" if quantity.get("uncertainty") is None else str(quantity["uncertainty"]),
                        " · ".join(str(origin.get(k) or "") for k in ("source_library", "library_version", "source_url")) or dataset["source"])
            add("Незавершено", r["unfinished_histories"], "историй")
            add("Время транспорта", r["timing"]["transport_seconds"], "с")
        else:
            info = r.get("run_info", {})
            add("Сценарий", "Поколения · legacy", source="legacy_unverified")
            add("Seed", info.get("seed"))
            add("Среднее k", r["k_mean"], "1", "Ошибка среднего не рассчитана", "Учебный показатель воспроизводства источника")
            add("Разброс поколений σ", r["k_std"], "1", "Стандартное отклонение, не SE")
            for name, value in info.items():
                if name not in ("seed", "material"): add(name, value, "см" if name == "radius_cm" else "")
            for name, value in info.get("material", {}).items():
                add(name, value, "барн" if name.startswith("sigma_") else "см" if name == "mean_free_path" else "", source="Исторические параметры; источник не подтверждён")
            for i, stats in enumerate(r["generation_stats"]):
                add(f"Поколение {i+1}: k", r["k_history"][i], "1", source=f"Обработано {stats['processed']}; исходы: {stats['outcome_counts']}")
        self.data_rows = rows
        if self.native:
            self.data_tree.delete(*self.data_tree.get_children())
            for row in rows: self.data_tree.insert("", "end", values=row)

    def _show_details(self):
        from tkinter import scrolledtext
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Ограничения и происхождение данных")
        dialog.geometry("720x520")
        text = scrolledtext.ScrolledText(dialog, wrap="word", padx=16, pady=16, font=("Segoe UI", 10))
        text.pack(fill="both", expand=True)
        details = self.warning_text or "Подробные сведения о данных появятся после расчёта."
        details += "\n\nПоколения: k_g = int(F_g × ν) / N_g до ограничения численности источника. σ по поколениям не является ошибкой среднего.\n\nСлой: постоянные коэффициенты; независимые истории. Интервалы — маргинальные биномиальные, без неопределённости параметров и ошибки модели."
        if self.experiment is not None and self.experiment["configuration"].get("reference_profile"):
            p = self.experiment["configuration"]["reference_profile"]
            details += (f"\n\n{p['source_library']} · {p['source_version']}\nЭнергетические границы: {p['energy_bounds_ev']} эВ\nМатериал: {p['material_temperature_K']} K\nСпектр: {p['spectrum']}\n"
                "Связанные состояния атомов, S(α,β), кристаллические эффекты и изменение энергии после столкновений не моделируются. "
                "Для группового профиля спектр удерживается фиксированным; ослабление с усреднённым коэффициентом не равно спектральному усреднению ослабления. "
                "Плотность графита задаётся пользователем и не проходит независимую проверку.\n\nИсточники:\n"+"\n".join(p["source_urls"]))
        text.insert("1.0", details)
        text.configure(state="disabled")

    def _select_generation(self, event=None):
        if self.result_scenario == "generations" and self.result and self.result["k_history"]:
            self.current_gen = int(self.generation_var.get())-1
            self._refresh_generation()

    def _reset_generation(self):
        self.current_gen = 0
        self._refresh_generation()

    def _step_gen(self, delta):
        if self.result_scenario != "generations" or not self.result or not self.result["k_history"]:
            return
        self.current_gen = (self.current_gen+delta) % len(self.result["k_history"])
        self._refresh_generation()

    def _refresh_generation(self):
        if not self.result or self.result_scenario != "generations":
            return
        self._update_status_full()
        if self.native:
            self.generation_var.set(str(self.current_gen+1))
            self._dirty_tabs.update(("graphs", "trajectories"))
            self._render_tab(self.active_tab)
        else:
            self._draw_trajectories()
            self._draw_k_chart()
            self._draw_bar()

    def _stop_animation(self):
        if self.anim_obj and self.anim_obj.event_source:
            self.anim_obj.event_source.stop()
        self.anim_obj = None
        self.anim_running = False
        if self.native:
            self.play_button.configure(text="Воспроизвести")

    def _on_anim(self, event=None):
        if self.result_scenario != "generations" or not self.result or not self.result["k_history"]:
            return
        if self.anim_running:
            self._stop_animation()
            return
        if not any(self.result.get("path_frames", [])):
            self._set_status("Нет сохранённых траекторий для воспроизведения.")
            return
        self.anim_running = True
        if self.native:
            self.play_button.configure(text="Пауза")
        def update(frame):
            self.current_gen = frame % len(self.result["k_history"])
            self._refresh_generation()
            return []
        self.anim_obj = animation.FuncAnimation(self.figures["trajectories"], update,
            frames=len(self.result["k_history"]), interval=650, repeat=True, blit=False, cache_frame_data=False)
        self.figures["trajectories"].canvas.draw_idle()

    def save_experiment(self, directory):
        if self.experiment is None:
            raise ValueError("Исследовательский экспорт JSON/CSV/NPZ доступен для завершённого эксперимента слоя.")
        return export_slab_experiment(self.experiment, directory)

    def open_experiment(self, directory):
        if self.worker.state == ExecutionState.RUNNING:
            raise RuntimeError("Дождитесь остановки текущего расчёта.")
        experiment = load_slab_experiment(directory)
        self._stop_animation()
        self.experiment, self.result = experiment, experiment["result"]
        self.result_config = deepcopy(experiment["configuration"])
        self.result_scenario = "slab"
        self.loaded_configuration = deepcopy(self.result_config)
        self._set("scenario", self.SCENARIOS[1])
        parameters = self.result_config["parameters"]
        for name, value in (("seed", parameters["seed"]), ("histories", parameters["n_histories"]), ("width", parameters["edges_cm"][-1]), ("max_collisions", parameters["max_collisions"])):
            self._set(name, value)
        self._set("material_mode", "Reference" if self.result_config.get("material_mode") == "reference" else "Учебная среда")
        profile = self.result_config.get("reference_profile")
        if profile:
            self._set("profile", profile["profile_id"])
            self._set("material", "Вода (H₂O)" if "water" in self.result_config["layers"][0]["material_id"] else "Графит")
            state = self.result_config["layers"][0].get("state") or {}
            origin = state.get("origin") or {}
            self._set("density", state.get("density") or "")
            self._set("density_sample", origin.get("original_reference") or "")
            self._set("density_method", origin.get("processing_description") or "")
        else:
            quantities = {q["name"]: q["values"] for q in self.result_config["layers"][0]["quantities"]}
            self._set("absorption", quantities.get("absorption_rate", ""))
            self._set("scattering", quantities.get("scattering_rate", ""))
        self._heat_cache = None
        if self.native:
            self._on_scenario_change()
            self.action_buttons["run"].configure(text="Повторить")
            self.export_menu.entryconfigure(0, state="normal")
            self.export_menu.entryconfigure(1, state="normal")
            self.notebook.select(self.pages["overview"])
        self._set_running_controls(False)
        self._render_all()
        self._set_status("Эксперимент загружен. «Повторить» использует полную исходную конфигурацию; для редактирования выберите «Новый».")
        return experiment

    def save_plot(self, path, *, tab=None):
        tab = tab or self.active_tab
        if tab not in self.figures:
            raise ValueError("Выберите вкладку с графиком для экспорта PNG.")
        if self.result is None:
            raise ValueError("Сначала выполните или откройте эксперимент.")
        if self.native:
            self._dirty_tabs.add(tab)
            self._render_tab(tab)
        self.figures[tab].savefig(path, dpi=300, bbox_inches="tight", facecolor="white")

    def _open_dialog(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(parent=self.root, title="Открыть сохранённый эксперимент слоя", filetypes=[("Манифест эксперимента", "experiment.json")])
        if path:
            try: self.open_experiment(Path(path).parent)
            except Exception as exc: messagebox.showerror("Не удалось открыть эксперимент", str(exc), parent=self.root)

    def _export_dialog(self):
        from tkinter import filedialog, messagebox
        path = filedialog.asksaveasfilename(parent=self.root, title="Новый каталог эксперимента (без расширения)", initialfile="heliosmc-experiment")
        if path:
            try:
                digest = self.save_experiment(path)
                self._set_status(f"Эксперимент сохранён: {path} · SHA-256: {digest}")
            except Exception as exc:
                messagebox.showerror("Не удалось сохранить эксперимент", str(exc), parent=self.root)

    def _plot_dialog(self):
        from tkinter import filedialog, messagebox
        if self.active_tab == "data":
            messagebox.showinfo("Экспорт графика", "Выберите вкладку с графиком.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Сохранить график", defaultextension=".png", filetypes=[("Изображение PNG", "*.png")])
        if path:
            try:
                self.save_plot(path)
                self._set_status(f"График сохранён: {path}")
            except Exception as exc:
                messagebox.showerror("Не удалось сохранить график", str(exc), parent=self.root)

    def new_experiment(self):
        if self.worker.state == ExecutionState.RUNNING:
            return
        self._stop_animation()
        self.loaded_configuration = None
        self.result = self.experiment = self.result_config = None
        self.result_radius = None
        self.result_scenario, self.current_gen, self.seed = "generations", 0, None
        for name, value in self.DEFAULTS.items(): self._set(name, value)
        self._heat_cache, self.data_rows = None, []
        self._set_running_controls(False)
        self._draw_empty()
        self.warning_text = "Legacy: исторические параметры; физическая валидация показателя не выполнена."
        if self.native:
            self._on_scenario_change()
            self.warning_var.set(self.warning_text)
            self.action_buttons["run"].configure(text="Запустить")
            self.export_menu.entryconfigure(0, state="disabled")
            self.export_menu.entryconfigure(1, state="disabled")
            self.notebook.select(self.pages["overview"])
        self._set_status("Готово к новому эксперименту")

    def _on_close(self, event=None):
        if self._closed:
            return
        self._closed = True
        self.worker.cancel()
        self._stop_animation()
        self._timer.stop()
        if self.native and self._after_id is not None:
            self.root.after_cancel(self._after_id)
        for figure in self.figures.values():
            plt.close(figure)
            figure.clear()
        if self.native:
            self.root.destroy()
