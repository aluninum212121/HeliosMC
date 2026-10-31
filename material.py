"""
БЛОК 2 — МАТЕРИАЛЫ.

Класс Material и библиотека предустановленных материалов MATERIALS.
Соответствует БЛОКУ 2 исходного монолита.
"""


import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransportData:
    """Constant, one-group inputs resolved before entering the transport kernel.

    Cross sections are microscopic barns (absorption excludes fission), the
    mean free path is in cm, and nu is dimensionless. All channels must use the
    same material state and normalization. No energy/temperature lookup occurs
    here. A future data provider must explicitly reduce its data to this model;
    passing tables to the current kernel is not supported.

    data_source is a caller-supplied identifier, not verified provenance.
    """

    sigma_f: float
    sigma_a: float
    sigma_s: float
    nu: float
    mean_free_path: float
    name: str = "unspecified"
    data_source: str = "unspecified"

    def __post_init__(self):
        for field in ("sigma_f", "sigma_a", "sigma_s", "nu", "mean_free_path"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")
            object.__setattr__(self, field, value)
        total = self.sigma_f + self.sigma_a + self.sigma_s
        if not math.isfinite(total) or total <= 0 or self.mean_free_path <= 0:
            raise ValueError("total cross section and mean_free_path must be positive and finite")
        if not isinstance(self.name, str) or not isinstance(self.data_source, str):
            raise TypeError("name and data_source must be strings")


def resolve_transport_data(material) -> TransportData:
    """Accept a snapshot, a get_transport_data() provider, or the legacy API.

    Providers resolve/validate their source outside Numba and return TransportData.
    The legacy validate()/mean_free_path()/scalar-field contract stays supported.
    """
    if isinstance(material, TransportData):
        return material
    provider = getattr(material, "get_transport_data", None)
    if provider is not None:
        data = provider()
        if not isinstance(data, TransportData):
            raise TypeError("get_transport_data() must return TransportData")
        return data
    material.validate()
    return TransportData(material.sigma_f, material.sigma_a, material.sigma_s,
                         material.nu, material.mean_free_path(),
                         name=getattr(material, "name", "unspecified"),
                         data_source="legacy_material")


class Material:
    """Existing constant presets: barns, g/cm^3, g/mol, cm and dimensionless nu.

    Preset values have no validated energy, temperature or evaluation provenance.
    """
    def __init__(self, name, sigma_f, sigma_a, sigma_s, nu, density, molar_mass, color):
        self.name       = name
        self.sigma_f    = sigma_f
        self.sigma_a    = sigma_a
        self.sigma_s    = sigma_s
        self.nu         = nu
        self.density    = density
        self.molar_mass = molar_mass
        self.color      = color        # цвет для визуализации

    def validate(self):
        for name in ("sigma_f", "sigma_a", "sigma_s", "nu", "density", "molar_mass"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.density <= 0 or self.molar_mass <= 0 or not math.isfinite(self.sigma_total()) or self.sigma_total() <= 0:
            raise ValueError("density, molar_mass and total cross section must be positive")

    def sigma_total(self):
        # ИСПРАВЛЕНИЕ: Теперь полное сечение включает деление (sigma_f)
        return self.sigma_f + self.sigma_a + self.sigma_s

    def macro_total(self):
        self.validate()
        N = (6.022e23 * self.density) / self.molar_mass
        total = N * self.sigma_total() * 1e-24
        if not math.isfinite(total) or total <= 0:
            raise ValueError("macroscopic total must be finite and positive")
        return total

    def mean_free_path(self):
        value = 1.0 / self.macro_total()
        if not math.isfinite(value) or value <= 0:
            raise ValueError("mean free path must be finite and positive")
        return value

    def get_transport_data(self) -> TransportData:
        """Freeze the existing formulas and parameters for one simulation run."""
        return TransportData(self.sigma_f, self.sigma_a, self.sigma_s, self.nu,
                             self.mean_free_path(), name=self.name,
                             data_source="constant_parameters")


MATERIALS = {
    "U-235 (чистый)":  Material("U-235",  sigma_f=1.65,  sigma_a=2.71,  sigma_s=8.9,  nu=2.43, density=19.1, molar_mass=235, color="#F5A623"),
    "U-235 (3.5%)":    Material("U-235 обогащ.", sigma_f=0.06, sigma_a=0.45, sigma_s=9.5, nu=2.43, density=10.5, molar_mass=238, color="#E8742A"),
    "Вода (H₂O)":      Material("H₂O",    sigma_f=0,     sigma_a=0.66,  sigma_s=49,   nu=0,    density=1.0,  molar_mass=18,  color="#4FC3F7"),
    "Графит":          Material("Графит",  sigma_f=0,     sigma_a=0.004, sigma_s=4.7,  nu=0,    density=1.7,  molar_mass=12,  color="#78909C"),
}
