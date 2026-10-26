"""
БЛОК 0 — ОБЩИЙ: совместимость с Numba.

Единая точка истины для HAS_NUMBA и декоратора-фолбэка njit,
чтобы остальные ветки (physics.py, gui.py) не дублировали этот код.
"""

# Попытка импорта Numba; если не установлена, используем Python-версию
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(func=None, **kwargs):
        return func if func else lambda f: f
