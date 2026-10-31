"""
БЛОК 2 — МАТЕРИАЛЫ.

Material, встроенные материалы и проверяемая схема структурированных данных.
Соответствует БЛОКУ 2 исходного монолита.
"""


import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


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


_QUANTITY_UNITS = {
    "sigma_f": "barn", "sigma_a": "barn", "sigma_s": "barn", "nu": "1",
    "mean_free_path": "cm", "density": "g/cm^3", "molar_mass": "g/mol",
}
_TRANSPORT_QUANTITIES = frozenset(_QUANTITY_UNITS)
_QUANTITY_UNITS.update(absorption_rate="1/cm", scattering_rate="1/cm")


def _text(field, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _nonnegative(field, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number, not a boolean or string")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return value


def _readonly_vector(field, values):
    if np.ma.isMaskedArray(values):
        raise TypeError(f"{field}: masked arrays are unsupported; use explicit missing status")
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real numeric NumPy array")
    if values.ndim != 1 or not values.size:
        raise ValueError(f"{field} must be a nonempty one-dimensional array")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{field} must contain finite nonnegative values")
    # An immutable bytes owner prevents both alias mutation and setflags(write=True).
    return np.frombuffer(values.tobytes(), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class MaterialQuantity:
    """One scalar or pointwise quantity; missing is never a fabricated zero.

    sigma_a means non-fission absorption. Other channel names may be stored but
    cannot silently add reactions to the current transport model. Arrays are
    copied into immutable float64 storage; equality of datasets is not inferred.
    """

    name: str
    unit: str | None
    values: float | np.ndarray | None = None
    status: str = "available"
    description: str = ""

    def __post_init__(self):
        _text("quantity.name", self.name)
        _text("quantity.status", self.status)
        if self.status not in ("available", "missing", "unsupported"):
            raise ValueError("quantity.status must be available, missing or unsupported")
        if not isinstance(self.description, str):
            raise TypeError("quantity.description must be a string")
        if self.unit is not None:
            _text("quantity.unit", self.unit)
            if self.unit not in ("barn", "1", "cm", "1/cm", "g/cm^3", "g/mol"):
                raise ValueError(f"{self.name}: unsupported unit {self.unit!r}; no implicit conversion")
            expected = _QUANTITY_UNITS.get(self.name)
            if expected is not None and self.unit != expected:
                raise ValueError(f"{self.name}: unit must be {expected}")
        if self.status != "available":
            if self.values is not None:
                raise ValueError(f"{self.name}: {self.status} quantities must have values=None")
            return
        if self.unit is None or self.values is None:
            raise ValueError(f"{self.name}: available quantities require unit and values")
        value = (_readonly_vector(self.name, self.values)
                 if isinstance(self.values, np.ndarray)
                 else _nonnegative(self.name, self.values))
        object.__setattr__(self, "values", value)


@dataclass(frozen=True, eq=False)
class MaterialDataset:
    """Scientific-data container, independent of storage format and transport.

    one_group stores constants; pointwise stores values on an explicit eV grid
    but is not executable by the current kernel. Provenance/classification are
    declarations, not evidence of physical validation. None denotes unknown
    composition, temperature or source version; those values are not guessed.
    """

    material_id: str
    name: str
    data_kind: str
    source: str
    version: str | None
    provenance: tuple[str, ...]
    quantities: tuple[MaterialQuantity, ...]
    composition: tuple[tuple[str, float], ...] | None = None
    composition_basis: str = "unknown"
    temperature: float | None = None
    temperature_unit: str = "K"
    energy_representation: str = "one_group"
    energy_grid: np.ndarray | None = None
    energy_unit: str = "eV"

    def __post_init__(self):
        for field in ("material_id", "name", "source", "data_kind", "temperature_unit",
                      "energy_unit", "energy_representation", "composition_basis"):
            _text(field, getattr(self, field))
        if any(c.isspace() for c in self.material_id):
            raise ValueError("material_id must not contain whitespace")
        if self.data_kind not in ("educational", "synthetic", "experimental", "evaluated"):
            raise ValueError("data_kind must be educational, synthetic, experimental or evaluated")
        if self.version is not None:
            _text("version", self.version)
        if not isinstance(self.provenance, (tuple, list)) or not self.provenance:
            raise ValueError("provenance must be a nonempty sequence of source/processing notes")
        for note in self.provenance:
            _text("provenance entry", note)
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if self.temperature_unit != "K" or self.energy_unit != "eV":
            raise ValueError("temperature_unit must be K and energy_unit must be eV; convert explicitly")
        if self.temperature is not None:
            object.__setattr__(self, "temperature", _nonnegative("temperature", self.temperature))
        self._validate_composition()
        if not isinstance(self.quantities, (tuple, list)) or not self.quantities:
            raise ValueError("quantities must be a nonempty sequence of MaterialQuantity")
        if any(not isinstance(q, MaterialQuantity) for q in self.quantities):
            raise TypeError("quantities entries must be MaterialQuantity")
        if len({q.name for q in self.quantities}) != len(self.quantities):
            raise ValueError("quantity names must be unique within a dataset")
        object.__setattr__(self, "quantities", tuple(self.quantities))
        if self.energy_representation == "one_group":
            if self.energy_grid is not None:
                raise ValueError("one_group must have energy_grid=None; no energy lookup is implied")
            if any(isinstance(q.values, np.ndarray) for q in self.quantities):
                raise ValueError("one_group quantities must be scalar")
        elif self.energy_representation == "pointwise":
            grid = _readonly_vector("energy_grid", self.energy_grid)
            if not np.all(np.diff(grid) > 0):
                raise ValueError("energy_grid must be strictly increasing with no duplicate energies")
            object.__setattr__(self, "energy_grid", grid)
            for q in self.quantities:
                if q.status == "available" and (not isinstance(q.values, np.ndarray)
                                                or q.values.shape != grid.shape):
                    raise ValueError(f"{q.name}: pointwise values must match energy_grid shape {grid.shape}")
        else:
            raise ValueError("energy_representation must be one_group or pointwise")

    def _validate_composition(self):
        bases = ("unknown", "atom_fraction", "mass_fraction", "stoichiometric")
        if self.composition_basis not in bases:
            raise ValueError(f"composition_basis must be one of {bases}")
        if self.composition is None:
            if self.composition_basis != "unknown":
                raise ValueError("missing composition requires composition_basis=unknown")
            return
        if self.composition_basis == "unknown" or not isinstance(self.composition, (tuple, list)) or not self.composition:
            raise ValueError("composition requires nonempty component/value pairs and an explicit basis")
        components = []
        for entry in self.composition:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("composition entries must be (component, amount) pairs")
            component, amount = entry
            _text("composition component", component)
            components.append((component, _nonnegative("composition amount", amount)))
        if len({name for name, _ in components}) != len(components):
            raise ValueError("composition components must be unique")
        try:
            total = math.fsum(amount for _, amount in components)
        except OverflowError:
            raise ValueError("composition amounts must have a positive finite sum") from None
        if total <= 0 or not math.isfinite(total):
            raise ValueError("composition amounts must have a positive finite sum")
        if self.composition_basis.endswith("fraction") and not math.isclose(total, 1., rel_tol=0., abs_tol=1e-8):
            raise ValueError("composition fractions must sum to 1 within 1e-8; no automatic normalization")
        object.__setattr__(self, "composition", tuple(components))

    def get_transport_data(self) -> TransportData:
        """Adapt explicit one-group constants; never reduce/interpolate a table."""
        if self.energy_representation != "one_group":
            raise ValueError("pointwise datasets are storage-only; energy-dependent transport is not implemented")
        available = {q.name: q.values for q in self.quantities if q.status == "available"}
        required = ("sigma_f", "sigma_a", "sigma_s", "nu", "mean_free_path")
        missing = [name for name in required if name not in available]
        if missing:
            raise ValueError(f"one_group transport requires available quantities: {', '.join(missing)}")
        extra = sorted(available.keys() - _TRANSPORT_QUANTITIES)
        if extra:
            raise ValueError(f"current transport does not support available quantities: {', '.join(extra)}")
        source_id = f"{self.data_kind}:{self.material_id}@{self.version or 'unknown'}; source={self.source}"
        return TransportData(*(available[name] for name in required),
                             name=self.name, data_source=source_id)


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


_BUILTIN_MATERIAL_IDS = {
    "U-235 (чистый)": "heliosmc:u235-pure",
    "U-235 (3.5%)": "heliosmc:u235-3.5",
    "Вода (H₂O)": "heliosmc:water",
    "Графит": "heliosmc:graphite",
}


def get_builtin_dataset(material_key: str) -> MaterialDataset:
    """Describe current preset values without inventing scientific provenance.

    Read the mutable Material at call time so there is no stale numeric registry.
    The ID identifies the preset, not a content hash; archive values for replay.
    """
    if material_key not in _BUILTIN_MATERIAL_IDS:
        raise ValueError(f"unknown built-in material key: {material_key!r}")
    material = MATERIALS[material_key]
    data = material.get_transport_data()
    values = {name: getattr(data, name) for name in
              ("sigma_f", "sigma_a", "sigma_s", "nu", "mean_free_path")}
    values.update(density=material.density, molar_mass=material.molar_mass)
    return MaterialDataset(
        material_id=_BUILTIN_MATERIAL_IDS[material_key], name=material.name,
        data_kind="educational", source="HeliosMC built-in constants", version=None,
        provenance=("Copied from the existing project presets; primary scientific reference, "
                    "composition details, temperature and energy weighting are unknown.",),
        quantities=tuple(MaterialQuantity(name, _QUANTITY_UNITS[name], value)
                         for name, value in values.items()),
    )


def resolve_medium_rates(dataset: MaterialDataset) -> tuple[float, float]:
    """Abstract one-group attenuation/scattering rates in cm^-1, no reactions.

    Only explicitly synthetic/educational datasets are admitted. Rates are
    supplied directly, not inferred from nuclear cross sections or densities.
    Zero total rate is a valid vacuum. Each record describes one spatial layer.
    """
    if not isinstance(dataset, MaterialDataset):
        raise TypeError("each medium layer must be a MaterialDataset")
    if dataset.data_kind not in ("synthetic", "educational"):
        raise ValueError("abstract scenarios require explicitly synthetic or educational data")
    if dataset.energy_representation != "one_group":
        raise ValueError("abstract medium rates must be one_group constants")
    available = {q.name: q for q in dataset.quantities if q.status == "available"}
    required = {"absorption_rate", "scattering_rate"}
    if available.keys() != required:
        raise ValueError("abstract medium requires only available absorption_rate and scattering_rate")
    if any(q.unit != "1/cm" for q in available.values()):
        raise ValueError("abstract medium rates must use 1/cm")
    absorption, scattering = (available[name].values for name in ("absorption_rate", "scattering_rate"))
    if not math.isfinite(absorption + scattering):
        raise ValueError("total medium rate must be finite")
    return absorption, scattering


def interpolate_synthetic_quantity(dataset: MaterialDataset, name: str,
                                   energies_ev: np.ndarray) -> np.ndarray:
    """Shape-preserving PCHIP on a synthetic table, with no extrapolation.

    Output uses the quantity's original units. This helper is not called by
    transport and does not establish a nuclear-data interpolation policy.
    """
    from scipy.interpolate import PchipInterpolator

    if not isinstance(dataset, MaterialDataset) or dataset.data_kind != "synthetic":
        raise ValueError("interpolation is restricted to synthetic MaterialDataset records")
    if dataset.energy_representation != "pointwise" or len(dataset.energy_grid) < 2:
        raise ValueError("interpolation requires a pointwise grid with at least two nodes")
    matches = [q for q in dataset.quantities if q.name == name and q.status == "available"]
    if not matches:
        raise ValueError(f"quantity {name!r} is absent or unavailable")
    if not isinstance(energies_ev, np.ndarray) or energies_ev.dtype.kind not in "iuf" or np.ma.isMaskedArray(energies_ev):
        raise TypeError("energies_ev must be a real numeric NumPy array")
    queries = np.asarray(energies_ev, dtype=np.float64)
    if queries.ndim != 1 or not np.isfinite(queries).all():
        raise ValueError("energies_ev must be a finite one-dimensional array")
    grid = dataset.energy_grid
    if ((queries < grid[0]) | (queries > grid[-1])).any():
        raise ValueError("energies_ev lie outside the grid; extrapolation is disabled")
    values = PchipInterpolator(grid, matches[0].values, extrapolate=False)(queries)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("interpolation produced nonfinite or negative values")
    return values
