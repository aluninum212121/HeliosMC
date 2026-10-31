"""Official entry point for the modular HeliosMC application."""

from gui import NeutronSimGUI
from numba_compat import HAS_NUMBA


def main():
    print("NeutronMC - starting interface...")
    print("Physics: " + ("Numba JIT" if HAS_NUMBA else "NumPy/Python"))
    return NeutronSimGUI()


if __name__ == "__main__":
    main()
