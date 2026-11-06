"""
БЛОК 2 — МАТЕРИАЛЫ.

Material, встроенные материалы и проверяемая схема структурированных данных.
Соответствует БЛОКУ 2 исходного монолита.
"""


import math
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date, datetime, timezone
from numbers import Real
from pathlib import Path
from urllib.parse import urlsplit

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


_DATA_KINDS = ("educational", "synthetic", "legacy", "experimental", "evaluated", "derived", "reference")


def _notes(name, values):
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a sequence of strings")
    for value in values:
        _text(name, value)
    return tuple(values)


def _optional_text(name, value):
    if value is not None:
        _text(name, value)


def _url(name, value):
    _text(name, value)
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{name} must not contain whitespace or control characters")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{name} must be an absolute HTTPS URL without credentials")


def _sha256(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("checksum must be a lowercase 64-character SHA-256 hex digest")


def _iso_date(name, value):
    _text(name, value)
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} must be an ISO date YYYY-MM-DD") from None
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be an ISO date YYYY-MM-DD")


@dataclass(frozen=True)
class DataOrigin:
    """Declared lineage, never inferred from an official URL or a checksum.

    library_version identifies the external release; MaterialDataset.version
    identifies the local dataset revision. None means unknown, not verified.
    Processing must describe parent records/versions and any derivation.
    """

    source_library: str | None = None
    library_version: str | None = None
    source_url: str | None = None
    original_reference: str | None = None
    processing_description: str | None = None
    processing_software: str | None = None
    processing_version: str | None = None
    file_checksum: str | None = None
    retrieval_date: str | None = None

    def __post_init__(self):
        for item in fields(self):
            _optional_text(item.name, getattr(self, item.name))
        if self.source_url is not None:
            _url("source_url", self.source_url)
        if self.file_checksum is not None:
            _sha256(self.file_checksum)
        if self.retrieval_date is not None:
            _iso_date("retrieval_date", self.retrieval_date)
        if self.library_version is not None and self.source_library is None:
            raise ValueError("library_version requires source_library")
        if self.processing_version is not None and self.processing_software is None:
            raise ValueError("processing_version requires processing_software")


@dataclass(frozen=True, eq=False)
class ParameterUncertainty:
    """Source-parameter standard uncertainty, not MC/numeric/ML error.

    Absolute uncertainty uses the parent quantity's unit; relative uncertainty
    is dimensionless (a fraction, not percent). No covariance or independence
    assumption is implied. The source/reference must describe the evidence.
    """

    values: float | np.ndarray
    kind: str
    reference: str

    def __post_init__(self):
        if self.kind not in ("absolute_standard", "relative_standard"):
            raise ValueError("uncertainty.kind must be absolute_standard or relative_standard")
        _text("uncertainty.reference", self.reference)
        value = (_readonly_vector("uncertainty", self.values)
                 if isinstance(self.values, np.ndarray) else _nonnegative("uncertainty", self.values))
        object.__setattr__(self, "values", value)


@dataclass(frozen=True)
class MaterialState:
    """Optional state metadata; no temperature law or density conversion.

    Density zero can describe an abstract vacuum. Ranges describe applicability
    in g/cm^3 and K, not inferred experimental confidence intervals.
    """

    density: float | None = None
    density_unit: str = "g/cm^3"
    physical_state: str = "unknown"
    origin: DataOrigin | None = None
    density_range: tuple[float, float] | None = None
    temperature_range: tuple[float, float] | None = None
    pressure: float | None = None
    pressure_unit: str = "Pa"

    def __post_init__(self):
        if self.density_unit != "g/cm^3":
            raise ValueError("density_unit must be g/cm^3; convert explicitly")
        if self.pressure_unit != "Pa":
            raise ValueError("pressure_unit must be Pa; convert explicitly")
        if self.pressure is not None:
            object.__setattr__(self, "pressure", _nonnegative("pressure", self.pressure))
        if self.physical_state not in ("unknown", "solid", "liquid", "gas", "plasma", "abstract"):
            raise ValueError("unsupported physical_state")
        if self.origin is not None and not isinstance(self.origin, DataOrigin):
            raise TypeError("state.origin must be DataOrigin or None")
        if self.density is not None:
            object.__setattr__(self, "density", _nonnegative("density", self.density))
        for name in ("density_range", "temperature_range"):
            limits = getattr(self, name)
            if limits is not None:
                if not isinstance(limits, (tuple, list)) or len(limits) != 2:
                    raise ValueError(f"{name} must contain lower and upper limits")
                lo, hi = (_nonnegative(name, x) for x in limits)
                if lo > hi:
                    raise ValueError(f"{name} lower limit exceeds upper limit")
                object.__setattr__(self, name, (lo, hi))
        if self.density is not None and self.density_range is not None:
            if not self.density_range[0] <= self.density <= self.density_range[1]:
                raise ValueError("density is outside density_range")


@dataclass(frozen=True)
class DataQuality:
    """A documented assessment declaration, not an automated validation proof.

    Scientific review is scoped by reference and applicability; the JSON reader
    only establishes schema validity and cannot grant scientific validation.
    """

    validation_status: str = "unverified"
    validation_reference: str | None = None
    applicability_notes: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()
    verification_status: str = "unverified"

    def __post_init__(self):
        if self.verification_status not in ("unverified", "source_checked", "legacy_unverified"):
            raise ValueError("unsupported verification_status")
        if self.validation_status not in ("unverified", "schema_validated", "scientifically_validated"):
            raise ValueError("unsupported validation_status")
        _optional_text("validation_reference", self.validation_reference)
        for name in ("applicability_notes", "known_limitations"):
            object.__setattr__(self, name, _notes(name, getattr(self, name)))
        if self.validation_status == "scientifically_validated":
            if self.validation_reference is None or not self.applicability_notes:
                raise ValueError("scientific validation requires a reference and applicability scope")


class UnverifiedDataWarning(UserWarning):
    """A requested scientific reference lacks documented validation."""


@dataclass(frozen=True)
class AveragingConditions:
    """Storage-only contract for a published average; never computes rates.

    State/composition belong to the enclosing dataset. Weighting must identify
    the spectrum and normalization, not just say 'thermal' or 'one group'.
    """
    method: str
    weighting: str
    energy_bounds_ev: tuple[float, float]
    reference: str
    assumptions: tuple[str, ...]
    independent_verification: str = "unverified"
    verification_reference: str | None = None

    def __post_init__(self):
        for name in ("method", "weighting", "reference"):
            _text(name, getattr(self, name))
        if not isinstance(self.energy_bounds_ev, (tuple, list)) or len(self.energy_bounds_ev) != 2:
            raise ValueError("averaging requires two energy bounds in eV")
        lo, hi = (_nonnegative("energy bound", v) for v in self.energy_bounds_ev)
        if lo >= hi:
            raise ValueError("averaging energy bounds must increase")
        object.__setattr__(self, "energy_bounds_ev", (lo, hi))
        object.__setattr__(self, "assumptions", _notes("averaging assumptions", self.assumptions))
        if not self.assumptions:
            raise ValueError("averaging assumptions are required")
        if self.independent_verification not in ("unverified", "documented"):
            raise ValueError("unsupported independent_verification")
        _optional_text("verification_reference", self.verification_reference)
        if self.independent_verification == "documented" and self.verification_reference is None:
            raise ValueError("documented independent verification requires its own reference")


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
    data_kind: str | None = None
    origin: DataOrigin | None = None
    uncertainty: ParameterUncertainty | None = None
    applicability_notes: tuple[str, ...] = ()
    averaging: AveragingConditions | None = None

    def __post_init__(self):
        _text("quantity.name", self.name)
        _text("quantity.status", self.status)
        if self.status not in ("available", "missing", "unsupported"):
            raise ValueError("quantity.status must be available, missing or unsupported")
        if not isinstance(self.description, str):
            raise TypeError("quantity.description must be a string")
        if self.averaging is not None and not isinstance(self.averaging, AveragingConditions):
            raise TypeError("averaging must be AveragingConditions or None")
        if self.data_kind is not None and self.data_kind not in _DATA_KINDS:
            raise ValueError(f"quantity.data_kind must be one of {_DATA_KINDS}")
        if self.origin is not None and not isinstance(self.origin, DataOrigin):
            raise TypeError("quantity.origin must be DataOrigin or None")
        if self.data_kind == "derived" and (self.origin is None or self.origin.processing_description is None):
            raise ValueError("derived quantity requires processing_description and parent provenance")
        object.__setattr__(self, "applicability_notes", _notes("quantity.applicability_notes", self.applicability_notes))
        if self.uncertainty is not None and not isinstance(self.uncertainty, ParameterUncertainty):
            raise TypeError("quantity.uncertainty must be ParameterUncertainty or None")
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
            if self.uncertainty is not None:
                raise ValueError(f"{self.name}: unavailable quantities cannot have uncertainty values")
            return
        if self.unit is None or self.values is None:
            raise ValueError(f"{self.name}: available quantities require unit and values")
        value = (_readonly_vector(self.name, self.values)
                 if isinstance(self.values, np.ndarray)
                 else _nonnegative(self.name, self.values))
        object.__setattr__(self, "values", value)
        if self.name == "molar_mass" and np.any(np.asarray(value) <= 0):
            raise ValueError("molar_mass must be positive")
        if self.uncertainty is not None:
            if np.shape(self.uncertainty.values) != np.shape(value):
                raise ValueError(f"{self.name}: uncertainty must match values shape; no broadcasting")

    @property
    def representation(self):
        """Representation follows values; it cannot contradict their shape."""
        if self.status != "available":
            return "unavailable"
        return "pointwise" if isinstance(self.values, np.ndarray) else "scalar"


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
    chemical_formula: str | None = None
    material_category: str = "unspecified"
    state: MaterialState = field(default_factory=MaterialState)
    origin: DataOrigin = field(default_factory=DataOrigin)
    quality: DataQuality = field(default_factory=DataQuality)
    local_file: "LocalDataFile | None" = None

    def __post_init__(self):
        for field in ("material_id", "name", "source", "data_kind", "temperature_unit",
                      "energy_unit", "energy_representation", "composition_basis"):
            _text(field, getattr(self, field))
        if any(c.isspace() for c in self.material_id):
            raise ValueError("material_id must not contain whitespace")
        if self.data_kind not in _DATA_KINDS:
            raise ValueError(f"data_kind must be one of {_DATA_KINDS}")
        _optional_text("chemical_formula", self.chemical_formula)
        if self.material_category not in ("unspecified", "component", "compound", "mixture", "abstract", "legacy_preset"):
            raise ValueError("unsupported material_category")
        for name, expected in (("state", MaterialState), ("origin", DataOrigin), ("quality", DataQuality)):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if self.local_file is not None and not isinstance(self.local_file, LocalDataFile):
            raise TypeError("local_file must be LocalDataFile or None")
        if self.data_kind in ("legacy", "synthetic", "educational") and self.quality.validation_status == "scientifically_validated":
            raise ValueError("legacy/synthetic/educational data cannot be labelled scientifically validated")
        if self.quality.verification_status == "source_checked":
            if self.data_kind in ("legacy", "synthetic", "educational"):
                raise ValueError("legacy/synthetic/educational data cannot be source_checked")
            if (self.origin.source_url is None or self.origin.original_reference is None
                    or self.origin.retrieval_date is None or not self.quality.applicability_notes):
                raise ValueError("source_checked requires source URL, reference, date and applicability")
        if self.data_kind == "derived" and self.origin.processing_description is None:
            raise ValueError("derived datasets require processing_description and parent provenance")
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
            limits = self.state.temperature_range
            if limits is not None and not limits[0] <= self.temperature <= limits[1]:
                raise ValueError("temperature is outside temperature_range")
        self._validate_composition()
        if self.material_category == "component" and self.composition is not None and len(self.composition) != 1:
            raise ValueError("component material must not declare multiple composition components")
        if not isinstance(self.quantities, (tuple, list)) or not self.quantities:
            raise ValueError("quantities must be a nonempty sequence of MaterialQuantity")
        if any(not isinstance(q, MaterialQuantity) for q in self.quantities):
            raise TypeError("quantities entries must be MaterialQuantity")
        if len({q.name for q in self.quantities}) != len(self.quantities):
            raise ValueError("quantity names must be unique within a dataset")
        object.__setattr__(self, "quantities", tuple(self.quantities))
        if self.quality.verification_status == "source_checked":
            for q in self.quantities:
                if q.status == "available" and (q.data_kind in ("legacy", "synthetic", "educational")
                        or q.origin is None or q.origin.source_url is None
                        or q.origin.original_reference is None or q.origin.retrieval_date is None
                        or not q.applicability_notes):
                    raise ValueError("source_checked quantities require individual provenance and applicability")
        density = next((q for q in self.quantities if q.name == "density" and q.status == "available"), None)
        if density is not None and self.state.density is not None:
            if isinstance(density.values, np.ndarray) or density.values != self.state.density:
                raise ValueError("density quantity conflicts with state.density")
            if self.state.origin is not None and density.origin is not None and self.state.origin != density.origin:
                raise ValueError("density quantity conflicts with state provenance")
        if self.energy_representation in ("one_group", "not_applicable"):
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
            raise ValueError("energy_representation must be one_group, pointwise or not_applicable")
        if self.energy_representation == "not_applicable" and any(
                q.name in ("sigma_f", "sigma_a", "sigma_s", "nu", "mean_free_path",
                           "absorption_rate", "scattering_rate") and q.status == "available"
                for q in self.quantities):
            raise ValueError("interaction quantities require an explicit energy representation")
        if self.quality.validation_status == "scientifically_validated" and self.energy_representation == "one_group":
            nuclear_averages = [q for q in self.quantities if q.status == "available"
                                and q.name in ("sigma_f", "sigma_a", "sigma_s")]
            if nuclear_averages and (self.temperature is None or self.composition is None or any(
                    q.averaging is None or q.averaging.independent_verification != "documented"
                    for q in nuclear_averages)):
                raise ValueError("scientifically validated nuclear averages require state, composition and documented averaging review")

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
            if self.composition_basis == "stoichiometric" and amount == 0:
                raise ValueError("stoichiometric amounts must be positive")
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
            raise ValueError("non-one-group datasets are storage-only; no reference/pointwise transport mapping is implemented")
        self._require_compatible_kind(allow_legacy=True)
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

    def _require_compatible_kind(self, *, allow_legacy=False):
        allowed = ("synthetic", "educational", "legacy") if allow_legacy else ("synthetic", "educational")
        if self.data_kind not in allowed:
            raise ValueError("scientific datasets are storage-only; abstract scenarios require synthetic or educational data; "
                             "no reviewed scientific-to-transport mapping exists")
        forbidden = ("experimental", "evaluated", "reference", "legacy") if self.data_kind != "legacy" else ("experimental", "evaluated", "reference")
        def scientific_origin(origin):
            return origin is not None and (origin.source_library is not None or origin.source_url is not None)
        if scientific_origin(self.origin) or scientific_origin(self.state.origin) or any(
            q.data_kind in forbidden or scientific_origin(q.origin)
            for q in self.quantities
        ):
            raise ValueError("scientific/legacy source quantities are storage-only in a relabelled dataset; "
                             "relabelling does not enable transport")

    def require_scientific_reference(self):
        """Fail visibly when an unverified dataset is requested as an oracle.

        Metadata documents a review; callers must still inspect that review for
        their application. This never enables transport or verifies a publication.
        """
        missing_average = self.energy_representation == "one_group" and any(
            q.status == "available" and q.name in ("sigma_f", "sigma_a", "sigma_s")
            and (q.averaging is None or q.averaging.independent_verification != "documented")
            for q in self.quantities)
        if (missing_average or self.data_kind not in ("experimental", "evaluated", "derived")
                or self.quality.validation_status != "scientifically_validated"
                or self.origin.original_reference is None
                or any(q.data_kind in ("legacy", "synthetic", "educational") for q in self.quantities)):
            message = f"{self.material_id}: unverified data cannot be used as a validated scientific reference"
            warnings.warn(message, UnverifiedDataWarning, stacklevel=2)
            raise ValueError(message)
        return self


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
    data_kind = "legacy"
    validation_status = "unverified"
    verification_status = "legacy_unverified"

    def require_scientific_reference(self):
        message = f"{self.name}: legacy/unverified constants cannot be used as a validated scientific reference"
        warnings.warn(message, UnverifiedDataWarning, stacklevel=2)
        raise ValueError(message)

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
        data_kind="legacy", source="HeliosMC built-in constants (unverified)", version=None,
        provenance=("Copied from the existing project presets; primary scientific reference, "
                    "composition details, temperature and energy weighting are unknown.",),
        quantities=tuple(MaterialQuantity(name, _QUANTITY_UNITS[name], value,
            data_kind="derived" if name == "mean_free_path" else "legacy",
            origin=DataOrigin(processing_description=(
                "1 / ((6.022e23 * density / molar_mass) * (sigma_f + sigma_a + sigma_s) * 1e-24); "
                "derived from this legacy preset, not independently measured.") if name == "mean_free_path" else None))
            for name, value in values.items()),
        material_category="legacy_preset", state=MaterialState(density=material.density),
        quality=DataQuality(verification_status="legacy_unverified", known_limitations=(
            "Unknown primary source, temperature, composition and energy weighting.",
            "Preset names do not establish physical validity; not a scientific reference.",
            "Excluded from reference ML datasets; no reactor validation or calibration is implied.")),
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
    dataset._require_compatible_kind()
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
    dataset._require_compatible_kind()
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


SYNTHETIC_TABLE_SCHEMA = "heliosmc.synthetic-material.v1"
SOURCE_CATALOG_SCHEMA = "heliosmc.source-catalog.v1"
_MAX_JSON_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class LocalDataFile:
    """A check of exact local bytes, not evidence of download or scientific truth.

    An observed hash alone is NOT integrity verification. expected_sha256 must
    be accompanied by its independently obtained checksum_reference. Only the
    synthetic reader below establishes a supported schema_version.
    """

    path: str
    size_bytes: int
    sha256: str
    expected_sha256: str | None = None
    checksum_reference: str | None = None
    schema_version: str | None = None
    acquisition: DataOrigin | None = None

    def __post_init__(self):
        _text("local path", self.path)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        _sha256(self.sha256)
        if (self.expected_sha256 is None) != (self.checksum_reference is None):
            raise ValueError("expected_sha256 and checksum_reference must be supplied together")
        if self.expected_sha256 is not None:
            _sha256(self.expected_sha256)
            _text("checksum_reference", self.checksum_reference)
            if self.expected_sha256 != self.sha256:
                raise ValueError("SHA-256 checksum mismatch")
        if self.schema_version is not None and self.schema_version != SYNTHETIC_TABLE_SCHEMA:
            raise ValueError("unsupported locally validated schema_version")
        if self.acquisition is not None:
            if not isinstance(self.acquisition, DataOrigin):
                raise TypeError("acquisition must be DataOrigin or None")
            if self.acquisition.source_url is None or self.acquisition.retrieval_date is None:
                raise ValueError("download acquisition requires source_url and retrieval_date")

    @property
    def integrity_verified(self):
        return self.expected_sha256 is not None

    @property
    def schema_validated(self):
        return self.schema_version == SYNTHETIC_TABLE_SCHEMA

    @property
    def downloaded(self):
        """Recorded acquisition, supplied by caller; a local file alone is insufficient."""
        return self.acquisition is not None


@dataclass(frozen=True)
class DataLibrary:
    """Offline source registration. URLs are inert strings, never downloaded.

    A catalog entry identifies one release/processing recipe, not interchangeable
    data. local_files records actual byte checks separately from availability.
    Unknown licenses remain None; software licensing is not data licensing.
    """

    library_id: str
    name: str
    organization: str
    version: str
    source_url: str
    data_kind: str
    formats: tuple[str, ...]
    checked_on: str
    documentation_urls: tuple[str, ...] = ()
    license: str | None = None
    usage_notes: str = "Terms must be checked for the particular release before redistribution."
    availability_status: str = "known_source"
    local_files: tuple[LocalDataFile, ...] = ()

    def __post_init__(self):
        for name in ("library_id", "name", "organization", "version", "usage_notes"):
            _text(name, getattr(self, name))
        if any(c.isspace() for c in self.library_id):
            raise ValueError("library_id must not contain whitespace")
        _url("source_url", self.source_url)
        _iso_date("checked_on", self.checked_on)
        if self.data_kind not in ("evaluated", "experimental", "derived"):
            raise ValueError("library data_kind must be evaluated, experimental or derived")
        object.__setattr__(self, "formats", _notes("formats", self.formats))
        if not self.formats or len(set(self.formats)) != len(self.formats):
            raise ValueError("formats must be nonempty and unique")
        object.__setattr__(self, "documentation_urls", _notes("documentation_urls", self.documentation_urls))
        for url in self.documentation_urls:
            _url("documentation_url", url)
        _optional_text("license", self.license)
        if self.availability_status not in ("known_source", "unavailable", "not_checked"):
            raise ValueError("availability_status describes the source, not file/scientific validation")
        if not isinstance(self.local_files, (tuple, list)) or any(not isinstance(f, LocalDataFile) for f in self.local_files):
            raise TypeError("local_files must be a sequence of LocalDataFile checks")
        if len({f.path for f in self.local_files}) != len(self.local_files):
            raise ValueError("duplicate local file paths")
        object.__setattr__(self, "local_files", tuple(self.local_files))


def register_library(catalog: dict[str, DataLibrary], library: DataLibrary):
    """Register explicitly; duplicate IDs never silently overwrite a release."""
    if not isinstance(catalog, dict) or not isinstance(library, DataLibrary):
        raise TypeError("registration requires a dict and a DataLibrary")
    if library.library_id in catalog:
        raise ValueError(f"duplicate library_id: {library.library_id}")
    catalog[library.library_id] = library


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _read_local_json(path, *, expected_sha256=None, checksum_reference=None, max_bytes=_MAX_JSON_BYTES):
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    path = Path(path)
    try:
        with path.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"cannot read local JSON file {path}: {exc}") from exc
    if len(payload) > max_bytes:
        raise ValueError(f"local JSON file exceeds max_bytes={max_bytes}")
    record = LocalDataFile(str(path.resolve()), len(payload), hashlib.sha256(payload).hexdigest(),
                           expected_sha256, checksum_reference)
    try:
        document = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=_strict_object,
                              parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    return document, record


def inspect_local_file(path, *, expected_sha256=None, checksum_reference=None,
                       acquisition: DataOrigin | None = None) -> LocalDataFile:
    """Hash an existing regular file with bounded working memory, without parsing.

    Optional acquisition is a caller's recorded download evidence, not a network
    action. No schema is declared checked, even if the extension is recognized.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"local file does not exist or is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ValueError(f"cannot inspect local file {path}: {exc}") from exc
    return LocalDataFile(str(path.resolve()), size, digest.hexdigest(), expected_sha256,
                         checksum_reference, acquisition=acquisition)


def _object_fields(value, name, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(allowed)
    if missing or extra:
        raise ValueError(f"{name}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}")


def _metadata(cls, value):
    _object_fields(value, cls.__name__, (f.name for f in fields(cls)))
    return cls(**value)


def _json_vector(name, value):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    # Check before NumPy coercion: [true, 1] or ["1", 2] must not become numbers.
    for item in value:
        _nonnegative(name, item)
    return _readonly_vector(name, np.asarray(value))


def load_synthetic_table(path, *, expected_sha256=None, checksum_reference=None,
                         max_bytes=_MAX_JSON_BYTES) -> MaterialDataset:
    """Load a bounded local JSON fixture; no network, pickle or nuclear parser.

    Exact v1 envelope and fields are required. Only explicitly synthetic data
    can enter this reader; grid/quantity validation uses the existing schema.
    A verified hash checks byte integrity, not scientific provenance. Reading
    does not perform interpolation, reduction, normalization or unit conversion.
    """
    document, record = _read_local_json(path, expected_sha256=expected_sha256,
        checksum_reference=checksum_reference, max_bytes=max_bytes)
    _object_fields(document, "table", ("schema_version", "dataset"), ("schema_version", "dataset"))
    if document["schema_version"] != SYNTHETIC_TABLE_SCHEMA:
        raise ValueError(f"unsupported schema_version; expected {SYNTHETIC_TABLE_SCHEMA}")
    data = document["dataset"]
    _object_fields(data, "dataset", (f.name for f in fields(MaterialDataset) if f.name != "local_file"),
        ("material_id", "name", "data_kind", "source", "version", "provenance", "quantities",
         "energy_representation", "energy_unit"))
    if data["data_kind"] != "synthetic":
        raise ValueError("local table reader accepts explicitly synthetic data only")
    dataset = material_from_dict(data)
    dataset._require_compatible_kind()
    return replace(dataset, local_file=replace(record, schema_version=SYNTHETIC_TABLE_SCHEMA),
                   quality=replace(dataset.quality, validation_status="schema_validated"))


def material_from_dict(data) -> MaterialDataset:
    """Decode inert metadata; validate fields without granting scientific trust.

    Local-file checks, when present, describe prior inspections, not new I/O.
    The synthetic v1 reader remains restricted to synthetic records.
    """
    _object_fields(data, "dataset", (f.name for f in fields(MaterialDataset)),
                   ("material_id", "name", "data_kind", "source", "version", "provenance", "quantities"))
    data = dict(data)
    if data.get("local_file") is not None:
        local = dict(data["local_file"])
        if local.get("acquisition") is not None:
            local["acquisition"] = _metadata(DataOrigin, local["acquisition"])
        data["local_file"] = _metadata(LocalDataFile, local)
    for name, cls in (("origin", DataOrigin), ("quality", DataQuality), ("state", MaterialState)):
        if name in data:
            value = data[name]
            if name == "state" and isinstance(value, dict) and "origin" in value and value["origin"] is not None:
                value = dict(value, origin=_metadata(DataOrigin, value["origin"]))
            data[name] = _metadata(cls, value)
    if data.get("energy_grid") is not None:
        data["energy_grid"] = _json_vector("energy_grid", data["energy_grid"])
    if not isinstance(data["quantities"], list) or not data["quantities"]:
        raise ValueError("quantities must be a nonempty JSON array")
    quantities = []
    for raw in data["quantities"]:
        _object_fields(raw, "quantity", (f.name for f in fields(MaterialQuantity)), ("name", "unit", "status"))
        raw = dict(raw)
        if isinstance(raw.get("values"), list):
            raw["values"] = _json_vector("quantity.values", raw["values"])
        if raw.get("origin") is not None:
            raw["origin"] = _metadata(DataOrigin, raw["origin"])
        if raw.get("averaging") is not None:
            raw["averaging"] = _metadata(AveragingConditions, raw["averaging"])
        if raw.get("uncertainty") is not None:
            u = raw["uncertainty"]
            _object_fields(u, "uncertainty", ("values", "kind", "reference"), ("values", "kind", "reference"))
            u = dict(u)
            if isinstance(u["values"], list):
                u["values"] = _json_vector("uncertainty.values", u["values"])
            raw["uncertainty"] = ParameterUncertainty(**u)
        quantities.append(MaterialQuantity(**raw))
    data["quantities"] = tuple(quantities)
    return MaterialDataset(**data)


def load_source_catalog(path=None) -> dict[str, DataLibrary]:
    """Read versioned source metadata offline; never inspect linked/local files.

    This file cannot claim download, checksum or scientific-validation success.
    Attach separately produced LocalDataFile records explicitly after checking.
    """
    path = Path(__file__).parent / "docs" / "data_sources.json" if path is None else path
    document, _ = _read_local_json(path)
    _object_fields(document, "catalog", ("schema_version", "libraries"), ("schema_version", "libraries"))
    if document["schema_version"] != SOURCE_CATALOG_SCHEMA:
        raise ValueError(f"unsupported catalog schema_version; expected {SOURCE_CATALOG_SCHEMA}")
    if not isinstance(document["libraries"], list):
        raise ValueError("libraries must be a JSON array")
    result = {}
    for raw in document["libraries"]:
        _object_fields(raw, "library", (f.name for f in fields(DataLibrary) if f.name != "local_files"),
            ("library_id", "name", "organization", "version", "source_url", "data_kind", "formats", "checked_on"))
        register_library(result, DataLibrary(**raw))
    return result


MATERIAL_SCHEMA = "heliosmc.material.v2"


def get_reference_dataset(material_key: str) -> MaterialDataset:
    """Source-checked ordinary-material facts, NEVER a transport preset.

    Source checked means transcription and scope checked on 2026-11-06. It
    does not mean independent experimental validation. No nuclear data occur.
    Unknown graphite grade/state density remains missing. Historical presets
    are deliberately untouched, including their rounded Avogadro constant.
    """
    if material_key not in ("Вода (H₂O)", "Графит"):
        raise ValueError("reference records exist only for water and graphite")
    water = material_key == "Вода (H₂O)"
    cas = "C7732185&Mask=1" if water else "C7782425&Units=SI"
    origin = DataOrigin(source_library="NIST Chemistry WebBook, SRD 69",
        source_url=f"https://webbook.nist.gov/cgi/cbook.cgi?ID={cas}",
        original_reference=f"NIST Chemistry WebBook: {'Water' if water else 'Graphite'}, formula and molecular weight",
        retrieval_date="2026-11-06",
        processing_description="Published molecular weight represented as conventional molar mass in g/mol; no isotope-specific mass inferred.")
    scope = ("Chemical identity and conventional molar mass; isotopic fractions and impurities unspecified.",)
    molar = MaterialQuantity("molar_mass", "g/mol", 18.0153 if water else 12.0107,
        data_kind="reference", origin=origin, applicability_notes=scope)
    if water:
        density_origin = DataOrigin(source_library="IAPWS", library_version="SR6-08(2011)",
            source_url="https://iapws.org/technical-guidance/release/LiquidWater.download",
            original_reference="Revised Supplementary Release on Properties of Liquid Water at 0.1 MPa, Table 8, p. 11",
            retrieval_date="2026-11-06",
            processing_description="Published correlation check value 997.047013 kg/m^3 divided by 1000 to g/cm^3; no EOS implemented.")
        density = MaterialQuantity("density", "g/cm^3", .997047013, data_kind="reference",
            origin=density_origin, applicability_notes=("Ordinary liquid water at 298.15 K and 100000 Pa only.",
                "Correlation check value, not a new measurement. Printed digits exceed physical uncertainty.",
                "Section 2.3 quotes relative 0.0001 percent with coverage factor k=2; not stored as a one-sigma uncertainty."))
        state = MaterialState(density.values, physical_state="liquid", origin=density_origin,
                              temperature_range=(298.15, 298.15), pressure=100000.)
    else:
        density = MaterialQuantity("density", "g/cm^3", status="missing", data_kind="reference",
            description="No grade, porosity, temperature or pressure identified for the historical graphite.",
            applicability_notes=("NIST X-Ray Mass Attenuation Table 1 assumes 1.7 g/cm^3; this is not a measured density for an identified sample.",))
        state = MaterialState(physical_state="solid")
    return MaterialDataset(
        material_id="heliosmc:reference-water" if water else "heliosmc:reference-graphite",
        name="Water reference" if water else "Graphite reference", data_kind="reference",
        source="NIST SRD 69; IAPWS SR6-08(2011)" if water else "NIST SRD 69",
        version="1", provenance=("Official source pages inspected 2026-11-06; no nuclear evaluation files downloaded.",),
        quantities=(molar, density), chemical_formula="H2O" if water else "C",
        composition=(("H", 2.), ("O", 1.)) if water else (("C", 1.),),
        composition_basis="stoichiometric", material_category="compound" if water else "component",
        temperature=298.15 if water else None, energy_representation="not_applicable",
        state=state, origin=origin, quality=DataQuality(validation_status="schema_validated",
            verification_status="source_checked", applicability_notes=scope + density.applicability_notes,
            known_limitations=("No measured sample composition, neutron interaction coefficients or independent material validation.",
                "Source-parameter standard uncertainties are unknown unless explicitly supplied; None never means zero.",
                "Reference values do not replace historical transport parameters.")))


def _json_data(value):
    """Explicit JSON types only; never serialize arbitrary Python objects."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
            raise ValueError("only finite numeric arrays may be exported")
        return value.tolist()
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("JSON keys must be strings")
        return {k: _json_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_data(v) for v in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("unsupported or nonfinite JSON value")


def material_to_dict(dataset: MaterialDataset) -> dict:
    """Lossless JSON-compatible schema representation, including explicit nulls."""
    if not isinstance(dataset, MaterialDataset):
        raise TypeError("dataset must be MaterialDataset")
    return _json_data(asdict(dataset))


def _canonical_bytes(value):
    return json.dumps(_json_data(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _content_hash(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_archive(path, schema, payload):
    """Exclusive creation; embedded hash detects corruption, not authorship."""
    document = dict(schema_version=schema, exported_at=datetime.now(timezone.utc).isoformat(), payload=payload)
    document["content_sha256"] = _content_hash(document)
    raw = _canonical_bytes(document)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("archive exceeds JSON size limit")
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _load_archive(path, schema, *, expected_sha256=None):
    document, _ = _read_local_json(path, expected_sha256=expected_sha256,
        checksum_reference="caller-pinned export SHA-256" if expected_sha256 is not None else None)
    names = ("schema_version", "exported_at", "payload", "content_sha256")
    _object_fields(document, "archive", names, names)
    if document["schema_version"] != schema:
        raise ValueError("unsupported archive schema_version")
    _text("exported_at", document["exported_at"])
    stamp = datetime.fromisoformat(document["exported_at"])
    if stamp.utcoffset() is None:
        raise ValueError("exported_at requires a timezone")
    _sha256(document["content_sha256"])
    body = {k: v for k, v in document.items() if k != "content_sha256"}
    if _content_hash(body) != document["content_sha256"]:
        raise ValueError("archive content checksum mismatch")
    return document["payload"]


def _check_reference_identity(dataset):
    # Reserve these IDs against altered values, sources, statuses and conditions,
    # even if an editor recomputes the self-checksum. Other records remain declarations.
    for key in ("Вода (H₂O)", "Графит"):
        reference = get_reference_dataset(key)
        if dataset.material_id == reference.material_id:
            if material_to_dict(dataset) != material_to_dict(reference):
                raise ValueError("built-in reference identity/content mismatch")


def export_material_reference(dataset, path) -> str:
    """Archive any material classification without upgrading its trust status.

    Returns exact file SHA-256 for independent retention. No network access.
    """
    data = material_to_dict(dataset)
    _check_reference_identity(dataset)
    return _write_archive(path, MATERIAL_SCHEMA, data)


def load_material_reference(path, *, expected_sha256=None) -> MaterialDataset:
    dataset = material_from_dict(_load_archive(path, MATERIAL_SCHEMA, expected_sha256=expected_sha256))
    _check_reference_identity(dataset)
    return dataset
