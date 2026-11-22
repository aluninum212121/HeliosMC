"""Optional Numba acceleration with a compatible Python decorator fallback."""

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(func=None, **kwargs):
        return func if func else lambda f: f
