"""Schema/adapter checks; source transcription tests are not physics validation."""

import copy
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

import physics
from material import (
    MATERIALS, SYNTHETIC_TABLE_SCHEMA, SOURCE_CATALOG_SCHEMA,
    DataOrigin, DataQuality, LocalDataFile, MaterialDataset, MaterialQuantity,
    MaterialState, ParameterUncertainty, UnverifiedDataWarning,
    get_builtin_dataset, inspect_local_file, interpolate_synthetic_quantity, load_source_catalog,
    load_synthetic_table, register_library, resolve_medium_rates, resolve_transport_data,
)


@pytest.fixture
def table():
    return {
        "schema_version": SYNTHETIC_TABLE_SCHEMA,
        "dataset": {
            "material_id": "fixture:invented", "name": "Invented abstract response",
            "data_kind": "synthetic", "source": "Analytic software fixture; no measurements",
            "version": "1", "provenance": ["Invented nonnegative response f(E)=2E."],
            "energy_representation": "pointwise", "energy_grid": [0., 1., 2.], "energy_unit": "eV",
            "composition": [["abstract-A", .25], ["abstract-B", .75]], "composition_basis": "atom_fraction",
            "material_category": "abstract", "temperature": 300., "temperature_unit": "K",
            "state": {"density": 1., "physical_state": "abstract", "temperature_range": [290., 310.],
                      "origin": {"original_reference": "Invented state for software checks"}},
            "quantities": [{"name": "response", "unit": "1", "status": "available", "values": [0., 2., 4.]},
                           {"name": "missing_channel", "unit": None, "status": "missing"},
                           {"name": "future_channel", "unit": "barn", "status": "unsupported"}],
        },
    }


def write_table(tmp_path, document):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
    return path


def abstract_dataset():
    return MaterialDataset("fixture:rates", "Invented abstract rates", "synthetic", "Software fixture", "1",
        ("Direct mathematical rates; no nuclear-data conversion.",),
        (MaterialQuantity("absorption_rate", "1/cm", .5), MaterialQuantity("scattering_rate", "1/cm", 0.)))


def test_extended_schema_and_explicit_unknowns():
    dataset = replace(abstract_dataset(), chemical_formula="A2B", material_category="mixture",
        composition=(("invented-A", .2), ("invented-B", .8)), composition_basis="mass_fraction",
        temperature=300., state=MaterialState(1., physical_state="abstract", density_range=(.5, 2.), temperature_range=(250., 350.)),
        origin=DataOrigin(original_reference="Synthetic mathematical definition, not an experiment"))
    assert dataset.composition == (("invented-A", .2), ("invented-B", .8))
    assert dataset.state.density_unit == "g/cm^3"
    assert dataset.origin.retrieval_date is dataset.origin.source_library is None
    assert dataset.quantities[0].uncertainty is None
    assert dataset.quality.validation_status == "unverified"
    assert resolve_medium_rates(dataset) == (.5, 0.)
    unknown = abstract_dataset()
    assert unknown.composition is unknown.temperature is unknown.state.density is None


@pytest.mark.parametrize("state", [
    {"density": -1.}, {"density": np.inf}, {"density": True}, {"density": "1"},
    {"density_unit": "kg/m^3"}, {"physical_state": "imagined"}, {"origin": {}},
    {"density_range": [2., 1.]}, {"density_range": [-1., 1.]}, {"density_range": [1.]},
    {"density": 2., "density_range": [0., 1.]}, {"temperature_range": [0., np.nan]},
])
def test_state_invalid_values(state):
    with pytest.raises((ValueError, TypeError)):
        MaterialState(**state)


@pytest.mark.parametrize("changes", [
    {"temperature": -1.}, {"temperature": np.nan}, {"temperature": True},
    {"temperature_unit": "C"}, {"temperature": 301., "state": MaterialState(temperature_range=(0., 300.))},
    {"chemical_formula": ""}, {"state": {}}, {"origin": {}}, {"quality": {}},
    {"composition": (("A", .2),), "composition_basis": "atom_fraction"},
    {"composition": (("A", .5), ("A", .5)), "composition_basis": "mass_fraction"},
    {"composition": (("A", True),), "composition_basis": "stoichiometric"},
    {"composition": (("A", 1.), ("B", 1.)), "composition_basis": "stoichiometric", "material_category": "component"},
])
def test_extended_dataset_invalid_values(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(abstract_dataset(), **changes)


def test_state_density_duplicates_must_agree_and_zero_is_explicit_vacuum():
    q = (MaterialQuantity("density", "g/cm^3", 1.),)
    dataset = replace(abstract_dataset(), quantities=q, state=MaterialState(1.))
    assert dataset.state.density == dataset.quantities[0].values
    with pytest.raises(ValueError, match="conflicts"):
        replace(dataset, state=MaterialState(2.))
    assert MaterialState(0.).density == 0.
    assert replace(abstract_dataset(), temperature=0.).temperature == 0.


@pytest.mark.parametrize("changes", [
    {"source_url": "http://example.org"}, {"source_url": "https://user:secret@example.org"},
    {"source_url": "https://exam ple.org"}, {"source_url": "https://example.org/\npath"},
    {"file_checksum": "no hash"}, {"retrieval_date": "2026-02-30"}, {"retrieval_date": "20261031"},
    {"library_version": "1"}, {"processing_version": "1"}, {"original_reference": ""},
])
def test_origin_requires_explicit_well_formed_metadata(changes):
    with pytest.raises(ValueError):
        DataOrigin(**changes)


def test_derived_quantities_and_data_require_processing_description():
    origin = DataOrigin(processing_description="Twice fixture response; parent fixture:v1")
    q = MaterialQuantity("response", "1", 2., data_kind="derived", origin=origin)
    assert q.origin.processing_description.startswith("Twice")
    with pytest.raises(ValueError, match="processing_description"):
        replace(q, origin=None)
    with pytest.raises(ValueError, match="processing_description"):
        replace(abstract_dataset(), data_kind="derived")
    derived = replace(abstract_dataset(), data_kind="derived", origin=origin)
    with pytest.raises(ValueError, match="storage-only"):
        resolve_transport_data(derived)


def test_parameter_uncertainty_is_explicit_immutable_and_shape_checked():
    original = np.array([.1, .2])
    u = ParameterUncertainty(original, "absolute_standard", "Invented fixture uncertainties, not measured")
    q = MaterialQuantity("response", "1", np.array([1., 2.]), uncertainty=u)
    original[:] = 5.
    np.testing.assert_array_equal(q.uncertainty.values, [.1, .2])
    with pytest.raises(ValueError):
        u.values.setflags(write=True)
    with pytest.raises(ValueError, match="no broadcasting"):
        replace(q, uncertainty=ParameterUncertainty(.1, "relative_standard", "Invented fixture"))
    with pytest.raises(ValueError, match="unavailable"):
        MaterialQuantity("response", None, status="missing", uncertainty=u)
    with pytest.raises(ValueError):
        ParameterUncertainty(.1, "monte_carlo_error", "Wrong uncertainty type")
    with pytest.raises(ValueError):
        ParameterUncertainty(-.1, "absolute_standard", "Invalid fixture")
    with pytest.raises(ValueError):
        ParameterUncertainty(.1, "absolute_standard", "")
    assert q.representation == "pointwise"
    assert MaterialQuantity("response", "1", 1.).representation == "scalar"
    assert MaterialQuantity("response", None, status="missing").representation == "unavailable"


def test_builtin_values_preserved_and_unverified_use_fails_visibly():
    expected = [(1.65, 2.71, 8.9, 2.43, 19.1, 235), (.06, .45, 9.5, 2.43, 10.5, 238),
                (0., .66, 49., 0., 1., 18), (0., .004, 4.7, 0., 1.7, 12)]
    for (key, material), values in zip(MATERIALS.items(), expected):
        assert tuple(getattr(material, name) for name in ("sigma_f", "sigma_a", "sigma_s", "nu", "density", "molar_mass")) == values
        dataset = get_builtin_dataset(key)
        assert material.data_kind == dataset.data_kind == "legacy"
        assert material.validation_status == dataset.quality.validation_status == "unverified"
        assert dataset.temperature is dataset.composition is dataset.origin.original_reference is None
        assert dataset.state.density == material.density
        mfp = next(q for q in dataset.quantities if q.name == "mean_free_path")
        assert mfp.data_kind == "derived" and "legacy preset" in mfp.origin.processing_description
        for item in (material, dataset):
            with pytest.warns(UnverifiedDataWarning), pytest.raises(ValueError, match="scientific reference"):
                item.require_scientific_reference()
        old, new = material.get_transport_data(), resolve_transport_data(dataset)
        for name in ("sigma_f", "sigma_a", "sigma_s", "nu", "mean_free_path", "name"):
            assert getattr(old, name) == getattr(new, name)


def test_scientific_status_requires_scope_and_cannot_promote_legacy():
    with pytest.raises(ValueError, match="reference and applicability"):
        DataQuality("scientifically_validated")
    review = DataQuality("scientifically_validated", "Mock review marker, no real validation",
                         ("Mock scope for metadata unit test",))
    with pytest.raises(ValueError, match="cannot be labelled"):
        replace(get_builtin_dataset(next(iter(MATERIALS))), quality=review)
    declared = replace(abstract_dataset(), data_kind="evaluated", quality=review)
    with pytest.warns(UnverifiedDataWarning), pytest.raises(ValueError, match="unverified"):
        declared.require_scientific_reference()
    declared = replace(declared, origin=DataOrigin(original_reference="Mock primary reference"))
    assert declared.require_scientific_reference() is declared  # Checks declarations only, not the mock publication.
    with pytest.raises(ValueError, match="storage-only"):
        resolve_transport_data(declared)


@pytest.mark.parametrize("kind", ["experimental", "evaluated", "derived", "legacy"])
def test_abstract_rates_reject_scientific_and_legacy_data(kind):
    origin = DataOrigin(processing_description="Mock parent and processing description")
    dataset = replace(abstract_dataset(), data_kind=kind, origin=origin)
    with pytest.raises(ValueError, match="synthetic or educational"):
        resolve_medium_rates(dataset)


def test_no_scientific_channel_can_be_hidden_in_synthetic_dataset():
    dataset = abstract_dataset()
    q = replace(dataset.quantities[0], data_kind="evaluated")
    dataset = replace(dataset, quantities=(q, dataset.quantities[1]))
    with pytest.raises(ValueError, match="storage-only"):
        resolve_medium_rates(dataset)
    with pytest.raises(ValueError, match="storage-only"):
        resolve_transport_data(dataset)
    dataset = replace(abstract_dataset(), origin=DataOrigin(source_library="Official-source test marker"))
    with pytest.raises(ValueError, match="storage-only"):
        resolve_medium_rates(dataset)
    legacy = replace(abstract_dataset(), quantities=(replace(abstract_dataset().quantities[0], data_kind="legacy"),
                                                    abstract_dataset().quantities[1]))
    with pytest.raises(ValueError, match="relabelled"):
        resolve_medium_rates(legacy)


def test_read_table_hash_schema_units_missing_and_interpolation(tmp_path, table):
    path = write_table(tmp_path, table)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = load_synthetic_table(path, expected_sha256=digest, checksum_reference="Independent synthetic fixture manifest")
    assert dataset.local_file.integrity_verified and dataset.local_file.schema_validated
    assert dataset.local_file.sha256 == digest and dataset.local_file.size_bytes == path.stat().st_size
    assert dataset.quality.validation_status == "schema_validated"
    assert dataset.quantities[1].values is None and dataset.quantities[2].values is None
    np.testing.assert_array_equal(dataset.energy_grid, [0., 1., 2.])
    np.testing.assert_allclose(interpolate_synthetic_quantity(dataset, "response", np.array([.5, 1.5])), [1., 3.])
    with pytest.raises(ValueError, match="storage-only"):
        resolve_transport_data(dataset)
    with pytest.raises(ValueError, match="one_group"):
        resolve_medium_rates(dataset)
    assert not load_synthetic_table(path).local_file.integrity_verified
    with pytest.warns(UnverifiedDataWarning), pytest.raises(ValueError):
        dataset.require_scientific_reference()


@pytest.mark.parametrize("change,message", [
    (lambda d: d.update(schema_version="heliosmc.synthetic-material.v2"), "schema_version"),
    (lambda d: d["dataset"].pop("source"), "missing fields"),
    (lambda d: d["dataset"].update(unrecognized=1), "unknown fields"),
    (lambda d: d["dataset"].update(data_kind="evaluated"), "synthetic data only"),
    (lambda d: d["dataset"].update(energy_unit="MeV"), "energy_unit"),
    (lambda d: d["dataset"].update(energy_grid=[0., 1., 1.]), "strictly increasing"),
    (lambda d: d["dataset"].update(energy_grid=[0., 2., 1.]), "strictly increasing"),
    (lambda d: d["dataset"].update(energy_grid=[0., True, 2.]), "boolean"),
    (lambda d: d["dataset"].update(energy_grid=[0., "1", 2.]), "real number"),
    (lambda d: d["dataset"].update(energy_grid=[]), "nonempty"),
    (lambda d: d["dataset"]["quantities"][0].update(values=[1., 2.]), "match energy_grid"),
    (lambda d: d["dataset"]["quantities"][0].update(values=[1., 2., -1.]), "nonnegative"),
    (lambda d: d["dataset"]["quantities"][0].update(values=1.), "match energy_grid"),
    (lambda d: d["dataset"]["quantities"][0].update(values=[1., 2., 1e309]), "finite"),
    (lambda d: d["dataset"]["quantities"][0].update(unit="millibarn"), "unsupported unit"),
    (lambda d: d["dataset"]["quantities"][1].update(values=0.), "values=None"),
    (lambda d: d["dataset"]["quantities"].append(copy.deepcopy(d["dataset"]["quantities"][0])), "unique"),
    (lambda d: d["dataset"].update(local_file={"schema_version": SYNTHETIC_TABLE_SCHEMA}), "unknown fields"),
    (lambda d: d["dataset"].update(quality={"validation_status": "scientifically_validated"}), "reference and applicability"),
])
def test_reject_corrupt_or_incompatible_tables(tmp_path, table, change, message):
    change(table)
    path = tmp_path / "synthetic.json"
    # Deliberately allow malformed IEEE JSON tokens here to exercise reader rejection.
    path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=message):
        load_synthetic_table(path)


@pytest.mark.parametrize("payload", [b'{"schema_version":', b'{"x":1,"x":2}', b'\xff\xfe', b'[]', b'null'])
def test_invalid_json_and_utf8(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_synthetic_table(path)


def test_local_file_checksum_size_and_missing_file(tmp_path, table):
    path = write_table(tmp_path, table)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_synthetic_table(path, expected_sha256="0" * 64, checksum_reference="Independent fixture manifest")
    with pytest.raises(ValueError, match="supplied together"):
        load_synthetic_table(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        load_synthetic_table(path, max_bytes=10)
    with pytest.raises(ValueError, match="positive integer"):
        load_synthetic_table(path, max_bytes=True)
    with pytest.raises(ValueError, match="cannot read"):
        load_synthetic_table(tmp_path / "absent.json")


def test_table_uncertainties_and_state_are_read_without_invention(tmp_path, table):
    table["dataset"]["quantities"][0]["uncertainty"] = {
        "values": [.0, .1, .2], "kind": "absolute_standard", "reference": "Invented software fixture"}
    dataset = load_synthetic_table(write_table(tmp_path, table))
    np.testing.assert_array_equal(dataset.quantities[0].uncertainty.values, [.0, .1, .2])
    assert dataset.state.origin.original_reference.startswith("Invented")
    assert dataset.quantities[1].uncertainty is None
    table["dataset"]["quantities"][0]["uncertainty"]["values"] = [.1]
    with pytest.raises(ValueError, match="match values shape"):
        load_synthetic_table(write_table(tmp_path, table))


def test_loaded_abstract_rates_preserve_slab_counts_and_metadata(tmp_path):
    raw = asdict(abstract_dataset())
    raw.pop("local_file")
    path = write_table(tmp_path, {"schema_version": SYNTHETIC_TABLE_SCHEMA, "dataset": raw})
    loaded = load_synthetic_table(path)
    config = dict(edges_cm=[0., 2.], n_histories=300, seed=2026, max_collisions=1, max_samples=5)
    old = physics.simulate_slab([abstract_dataset()], **config)
    new = physics.simulate_slab([loaded], **config)
    assert new["outcome_counts"] == old["outcome_counts"]
    assert new["estimates"] == old["estimates"] and new["warnings"] == old["warnings"]
    for a, b in zip(old["sampled_paths_cm"], new["sampled_paths_cm"]):
        np.testing.assert_array_equal(a, b)
    assert new["run_info"]["layers"][0]["local_file"]["sha256"] == loaded.local_file.sha256


def test_offline_catalog_and_explicit_registration(tmp_path, monkeypatch):
    # Import/read must not initiate network requests, even for official URLs.
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("network used"))
    catalog = load_source_catalog()
    assert set(catalog) == {"endfb-viii.1", "jeff-4.0", "jendl-5", "openmc-endfb-viii.1-njoy2016.78",
        "exfor", "nist-ncnr-isotope-abundances-1992", "nist-srd69-reference-molar-masses", "iapws-sr6-08-2011-density",
        "jendl-4.0-carbon-profile-summaries"}
    original_size = len(catalog)
    for library in catalog.values():
        assert library.availability_status == "known_source" and library.local_files == ()
        assert library.source_url.startswith("https://")
    assert catalog["exfor"].data_kind == "experimental"
    assert catalog["openmc-endfb-viii.1-njoy2016.78"].data_kind == "derived"
    assert catalog["endfb-viii.1"].license is None
    first = next(iter(catalog.values()))
    with pytest.raises(ValueError, match="duplicate"):
        register_library(catalog, first)
    register_library(catalog, replace(first, library_id="fixture:registration", version="fixture-version"))
    assert len(catalog) == original_size + 1
    path = tmp_path / "catalog.json"
    document = {"schema_version": SOURCE_CATALOG_SCHEMA, "libraries": []}
    path.write_text(json.dumps(document), encoding="utf-8")
    assert load_source_catalog(path) == {}
    document["schema_version"] = "unknown"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_source_catalog(path)


@pytest.mark.parametrize("change", [
    {"availability_status": "scientifically_validated"}, {"source_url": "file:///secret"},
    {"formats": []}, {"checked_on": "yesterday"}, {"local_files": ["fake-file"]},
])
def test_library_status_and_file_checks_are_not_scientific_validation(change):
    library = next(iter(load_source_catalog().values()))
    with pytest.raises((ValueError, TypeError)):
        replace(library, **change)


def test_observed_hash_does_not_imply_integrity_schema_or_scientific_validation():
    record = LocalDataFile("fixture-local-only.json", 0, hashlib.sha256(b"").hexdigest())
    assert not record.integrity_verified and not record.schema_validated
    assert not record.downloaded
    library = replace(next(iter(load_source_catalog().values())), local_files=(record,))
    assert library.availability_status == "known_source"
    assert library.local_files[0] == record
    with pytest.raises(ValueError, match="unsupported locally"):
        replace(record, schema_version="ENDF-6")


def test_catalog_file_inspection_separates_acquisition_integrity_and_schema(tmp_path):
    path = tmp_path / "arbitrary.bin"
    path.write_bytes(b"Synthetic fixture, not scientific data or an ENDF file")
    raw = inspect_local_file(path)
    assert not raw.downloaded and not raw.integrity_verified and not raw.schema_validated
    # Test-only declaration: no download happens in this test or in inspect_local_file.
    acquisition = DataOrigin(source_url="https://example.org/fictional-fixture", retrieval_date="2026-10-31")
    checked = inspect_local_file(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checksum_reference="Independent fixture manifest", acquisition=acquisition)
    assert checked.downloaded and checked.integrity_verified and not checked.schema_validated
    with pytest.raises(ValueError, match="acquisition requires"):
        inspect_local_file(path, acquisition=DataOrigin(source_url="https://example.org"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect_local_file(path, expected_sha256="0" * 64, checksum_reference="Fixture manifest")
    with pytest.raises(ValueError, match="regular file"):
        inspect_local_file(tmp_path)


def test_reference_records_have_scoped_sources_without_replacing_presets():
    from material import get_reference_dataset
    water = get_reference_dataset("Вода (H₂O)")
    graphite = get_reference_dataset("Графит")
    assert water.state.density == 997.047013 / 1000
    assert water.temperature == 298.15 and water.state.pressure == 100000.
    assert water.composition == (("H", 2.), ("O", 1.))
    assert water.quantities[0].values == 18.0153
    assert graphite.quantities[0].values == 12.0107
    assert graphite.quantities[1].status == "missing"
    assert graphite.state.density is graphite.temperature is graphite.state.pressure is None
    assert MATERIALS["Вода (H₂O)"].density == 1.
    assert MATERIALS["Графит"].molar_mass == 12
    for dataset in (water, graphite):
        assert dataset.quality.verification_status == "source_checked"
        assert dataset.quality.validation_status != "scientifically_validated"
        assert dataset.energy_representation == "not_applicable"
        assert all(q.uncertainty is None for q in dataset.quantities)
        for q in dataset.quantities:
            if q.status == "available":
                assert q.origin.source_url and q.origin.retrieval_date == "2026-11-06"
                assert q.applicability_notes
        with pytest.raises(ValueError):
            resolve_transport_data(dataset)
        with pytest.raises(ValueError):
            resolve_medium_rates(dataset)
        with pytest.warns(UnverifiedDataWarning), pytest.raises(ValueError):
            dataset.require_scientific_reference()
    for key in list(MATERIALS)[:2]:
        assert get_builtin_dataset(key).quality.verification_status == "legacy_unverified"
        with pytest.raises(ValueError, match="only"):
            get_reference_dataset(key)


@pytest.mark.parametrize("state", [{"pressure": -1}, {"pressure": True}, {"pressure": float("inf")}, {"pressure_unit": "MPa"}])
def test_reference_pressure_validation(state):
    with pytest.raises((ValueError, TypeError)):
        MaterialState(**state)


def test_missing_molar_mass_composition_and_source_checks():
    from material import get_reference_dataset
    with pytest.raises(ValueError, match="positive"):
        MaterialQuantity("molar_mass", "g/mol", 0.)
    water = get_reference_dataset("Вода (H₂O)")
    with pytest.raises(ValueError, match="stoichiometric"):
        replace(water, composition=(("H", 0.), ("O", 1.)))
    with pytest.raises(ValueError, match="source_checked"):
        replace(water, origin=DataOrigin())
    with pytest.raises(ValueError, match="source_checked"):
        replace(water, data_kind="synthetic")
    # State provenance alone must not sneak scientific sources into a medium.
    disguised = replace(abstract_dataset(), state=MaterialState(origin=water.origin))
    with pytest.raises(ValueError, match="storage-only"):
        resolve_medium_rates(disguised)


def test_material_v2_roundtrip_and_pinned_reference(tmp_path):
    from material import (get_reference_dataset, export_material_reference, load_material_reference,
                          material_to_dict, _content_hash)
    for key in ("Вода (H₂O)", "Графит"):
        dataset = get_reference_dataset(key)
        path = tmp_path / (dataset.material_id.split(":")[-1] + ".json")
        checksum = export_material_reference(dataset, path)
        restored = load_material_reference(path, expected_sha256=checksum)
        assert material_to_dict(restored) == material_to_dict(dataset)
        with pytest.raises(FileExistsError):
            export_material_reference(dataset, path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["payload"]["quantities"][0]["values"] += 1
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="checksum"):
            load_material_reference(path)
        raw["content_sha256"] = _content_hash({k: v for k, v in raw.items() if k != "content_sha256"})
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="identity/content"):
            load_material_reference(path)
        with pytest.raises(ValueError, match="checksum"):
            load_material_reference(path, expected_sha256=checksum)


def test_material_archive_preserves_table_missing_uncertainty_and_legacy(tmp_path):
    from material import export_material_reference, load_material_reference, material_to_dict
    table = replace(abstract_dataset(), quantities=(MaterialQuantity("response", "1", np.array([1., 2.]),
        uncertainty=ParameterUncertainty(np.array([.1, .2]), "absolute_standard", "invented")),
        MaterialQuantity("other", None, status="missing")), energy_representation="pointwise", energy_grid=np.array([0., 2.]))
    for i, dataset in enumerate((table, get_builtin_dataset(next(iter(MATERIALS))))):
        path = tmp_path / f"{i}.json"
        export_material_reference(dataset, path)
        restored = load_material_reference(path)
        assert material_to_dict(restored) == material_to_dict(dataset)
    restored = load_material_reference(tmp_path / "0.json")
    with pytest.raises(ValueError):
        restored.energy_grid.setflags(write=True)
    with pytest.raises(ValueError):
        restored.quantities[0].uncertainty.values[0] = .4


def test_averaged_scientific_values_require_explicit_scope():
    from material import AveragingConditions, material_from_dict, material_to_dict
    origin = DataOrigin(original_reference="Test-only declaration")
    review = DataQuality("scientifically_validated", "Test-only review", ("Mock conditions",))
    dataset = replace(abstract_dataset(), data_kind="evaluated", origin=origin,
        quantities=(MaterialQuantity("sigma_s", "barn", 1.),))
    with pytest.raises(ValueError, match="documented averaging"):
        replace(dataset, quality=review)
    with pytest.warns(UnverifiedDataWarning), pytest.raises(ValueError):
        dataset.require_scientific_reference()
    average = AveragingConditions("flux-weighted integral", "Test-only uniform weight normalized on [0,1] eV",
        (0., 1.), "Test-only source", ("Mock fixed state",))
    dataset = replace(dataset, quantities=(replace(dataset.quantities[0], averaging=average),))
    with pytest.raises(ValueError, match="documented averaging"):
        replace(dataset, quality=review)
    with pytest.warns(UnverifiedDataWarning), pytest.raises(ValueError):
        dataset.require_scientific_reference()
    assert material_to_dict(material_from_dict(material_to_dict(dataset))) == material_to_dict(dataset)
    with pytest.raises(ValueError):
        replace(average, energy_bounds_ev=(1., 0.))
    with pytest.raises(ValueError):
        replace(average, assumptions=())
    with pytest.raises(ValueError, match="own reference"):
        replace(average, independent_verification="documented")
    checked = replace(average, independent_verification="documented", verification_reference="Mock independent review")
    dataset = replace(dataset, composition=(("X", 1.),), composition_basis="atom_fraction", temperature=300.,
                      quantities=(replace(dataset.quantities[0], averaging=checked),), quality=review)
    assert dataset.require_scientific_reference() is dataset  # Declaration only; no physical data used.


@pytest.mark.parametrize("temperature,kg_m3", [(260., 997.068360), (298.15, 997.047013), (375., 957.009710)])
def test_iapws_density_against_published_independent_check_points(temperature, kg_m3):
    from material import _water_density_0p1mpa
    # Table 8 printed resolution, fixed independently of implementation.
    assert 1000 * _water_density_0p1mpa(temperature) == pytest.approx(kg_m3, abs=5e-7, rel=0)


@pytest.mark.parametrize("temperature", [250., 400., True, np.inf, "300"])
def test_iapws_outside_contract_rejected(temperature):
    from material import _water_density_0p1mpa
    with pytest.raises((ValueError, TypeError)):
        _water_density_0p1mpa(temperature)


def reference_water():
    from material import get_reference_transport_dataset, REFERENCE_APPROXIMATION
    return get_reference_transport_dataset("Вода (H₂O)", approximation=REFERENCE_APPROXIMATION)


def test_reference_number_density_units_and_stoichiometry():
    from scipy.constants import Avogadro
    from material import resolve_reference_rates
    d = reference_water()
    q = {q.name: q for q in d.quantities}
    assert d.temperature == 300. and d.state.pressure == 100000.
    assert dict(d.composition) == {"H1": 1.9997, "H2": .0003, "O16": .99762, "O17": .00038, "O18": .002}
    # Independent transcription of the opened JAEA point columns; includes O17(n,alpha).
    sigma_a = 2*(.99985*.3326 + .00015*.0005500) + .99762*.0001899 + .00038*(.0006700+.2438) + .002*.0001408
    sigma_s = 2*(.99985*30.27 + .00015*4.246) + .99762*3.986 + .00038*3.615 + .002*3.532
    factor = d.state.density / 18.0153 * Avogadro * 1e-24
    np.testing.assert_allclose(resolve_reference_rates(d), factor*np.array([sigma_a, sigma_s]), rtol=2e-15)
    assert q["sigma_a_O17"].values == pytest.approx(.24447)
    assert q["sigma_a_H2"].values == .00055
    assert q["sigma_a"].averaging.energy_bounds_ev == (.0253, .0253)
    assert q["sigma_a"].averaging.method == "point_evaluation"
    assert q["sigma_a"].averaging.independent_verification == "unverified"
    assert all(q.origin.source_url and q.origin.retrieval_date for q in d.quantities)
    with pytest.raises(ValueError, match="storage-only"):
        d.get_transport_data()  # No path into the multiplying legacy engine.


def test_reference_graphite_explicit_density_and_atom_fraction_contract():
    from material import get_reference_transport_dataset, resolve_reference_rates, REFERENCE_APPROXIMATION
    from scipy.constants import Avogadro
    kwargs = dict(approximation=REFERENCE_APPROXIMATION)
    with pytest.raises(ValueError, match="requires explicit density"):
        get_reference_transport_dataset("Графит", **kwargs)
    origin = DataOrigin(original_reference="Invented sample G for software testing only",
        processing_description="User input 1.6 g/cm^3, 300 K; no measured material claimed")
    a = get_reference_transport_dataset("Графит", density=1.6, density_origin=origin, **kwargs)
    b = get_reference_transport_dataset("Графит", density=2.0, density_origin=origin, **kwargs)
    factor = 1.6 * Avogadro / 12.0107 * 1e-24
    np.testing.assert_allclose(resolve_reference_rates(a), factor*np.array([.989*.003877+.011*.001488, .989*4.950+.011*5.557]))
    np.testing.assert_allclose(resolve_reference_rates(b), np.array(resolve_reference_rates(a))*1.25)
    assert a.composition_basis == "atom_fraction" and sum(v for _, v in a.composition) == 1.
    assert a.quality.verification_status == "unverified"


@pytest.mark.parametrize("change", ["temperature", "energy", "composition", "mass_fraction", "source", "missing", "density", "id", "status"])
def test_reference_adapter_rejects_inconsistent_or_forged_inputs(change):
    from material import resolve_reference_rates
    d = reference_water()
    if change == "temperature": d = replace(d, temperature=301., state=replace(d.state, temperature_range=(300., 301.)))
    if change == "energy":
        d = replace(d, quantities=tuple(replace(q, averaging=replace(q.averaging, energy_bounds_ev=(.1,.1))) if q.averaging else q for q in d.quantities))
    if change == "composition": d = replace(d, composition=(("H1", 2.), ("O16", 1.)))
    if change == "mass_fraction": d = replace(d, composition=(("H1", .5), ("O16", .5)), composition_basis="mass_fraction")
    if change == "source": d = replace(d, origin=replace(d.origin, source_url="https://example.org/forged"))
    if change == "missing": d = replace(d, quantities=tuple(replace(q, status="missing", values=None) if q.name == "sigma_s_H1" else q for q in d.quantities))
    if change == "density":
        d = replace(d, state=replace(d.state, density=1.), quantities=tuple(replace(q, values=1.) if q.name == "density" else q for q in d.quantities))
    if change == "id": d = replace(d, material_id="custom:forged")
    if change == "status": d = replace(d, quality=replace(d.quality, verification_status="unverified"))
    with pytest.raises(ValueError, match="reference transport"):
        resolve_reference_rates(d)


def test_reference_roundtrip_and_mode_isolation(tmp_path):
    from material import (get_reference_transport_dataset, get_reference_dataset, REFERENCE_APPROXIMATION,
        export_material_reference, load_material_reference, material_to_dict, resolve_reference_rates)
    d = reference_water()
    digest = export_material_reference(d, tmp_path / "reference.json")
    restored = load_material_reference(tmp_path / "reference.json", expected_sha256=digest)
    assert material_to_dict(d) == material_to_dict(restored)
    assert resolve_medium_rates(restored, mode="reference") == resolve_reference_rates(d)
    for key in ("U-235 (чистый)", "U-235 (3.5%)"):
        with pytest.raises(ValueError, match="legacy_unverified"):
            get_reference_transport_dataset(key, approximation=REFERENCE_APPROXIMATION)
    with pytest.raises(ValueError, match="bound-atom"):
        get_reference_transport_dataset("Вода (H₂O)", approximation="bound_atom")
    with pytest.raises(ValueError): resolve_medium_rates(d)
    with pytest.raises(ValueError): resolve_medium_rates(abstract_dataset(), mode="reference")
    with pytest.raises(ValueError): resolve_medium_rates(get_reference_dataset("Вода (H₂O)"), mode="reference")
    with pytest.raises(ValueError): resolve_medium_rates(abstract_dataset(), mode="automatic")
    with pytest.raises(ValueError, match="unit must be barn"):
        MaterialQuantity("sigma_a_H1", "1/cm", .1)


def test_reserved_reference_transport_export_rejects_rehashed_tampering(tmp_path):
    from material import export_material_reference, load_material_reference, _content_hash
    export_material_reference(reference_water(), tmp_path / "ref.json")
    raw = json.loads((tmp_path / "ref.json").read_text(encoding="utf-8"))
    raw["payload"]["quantities"][-1]["values"] = 100.
    raw["content_sha256"] = _content_hash({k:v for k,v in raw.items() if k != "content_sha256"})
    (tmp_path / "ref.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="reference transport identity"):
        load_material_reference(tmp_path / "ref.json")


def graphite_profile(profile_id):
    from material import get_reference_profiles, get_reference_transport_dataset
    profile = get_reference_profiles()[profile_id]
    origin = DataOrigin(original_reference="Software fixture, not a measured graphite sample",
        processing_description="Caller chooses 1.6 g/cm^3 at 300 K")
    return get_reference_transport_dataset("Графит", profile_id=profile_id,
        approximation=profile["approximation"], density=1.6, density_origin=origin)


def test_reference_profile_catalog_is_fresh_scoped_and_explicit():
    from material import get_reference_profiles, REFERENCE_POINT_PROFILE, REFERENCE_CONTROL_PROFILE, REFERENCE_MAXWELL_PROFILE
    profiles = get_reference_profiles()
    assert set(profiles) == {REFERENCE_POINT_PROFILE, REFERENCE_CONTROL_PROFILE, REFERENCE_MAXWELL_PROFILE}
    for p in profiles.values():
        assert p["material_temperature_K"] == 300.
        assert p["numerical_verification_status"] == "source_checked"
        assert p["scientific_validation_status"] == "unverified"
        assert p["source_urls"] and p["source_version"] and p["limitations"]
    assert profiles[REFERENCE_POINT_PROFILE]["available_materials"] == ["Вода (H₂O)", "Графит"]
    group = profiles[REFERENCE_MAXWELL_PROFILE]
    assert group["energy_bounds_ev"] == [1e-5, 10.]
    assert group["spectrum_kind"] == "flux_average" and group["spectrum_kT_ev"] == .0253
    # Independently use the exact SI defining values of elementary charge/k_B.
    assert group["spectrum_temperature_K"] == pytest.approx(.0253*1.602176634e-19/1.380649e-23, rel=1e-12)
    assert "Вода (H₂O)" in group["unavailable_materials"]
    group["energy_bounds_ev"][0] = 0.
    assert get_reference_profiles()[REFERENCE_MAXWELL_PROFILE]["energy_bounds_ev"] == [1e-5, 10.]


def test_published_carbon_macs_is_converted_to_normalized_flux_mean():
    from material import REFERENCE_MAXWELL_PROFILE, resolve_reference_rates
    from scipy.constants import Avogadro
    d = graphite_profile(REFERENCE_MAXWELL_PROFILE)
    q = {q.name:q for q in d.quantities}
    # Independent source transcriptions, not group values treated as point data.
    assert q["published_macs_sigma_a_C0"].values == .003862
    assert q["published_macs_sigma_s_C0"].values == 5.575
    factor = np.sqrt(np.pi)/2
    assert q["sigma_a_C0"].values == pytest.approx(.003862*factor, rel=1e-15)
    assert q["sigma_s_C0"].values == pytest.approx(5.575*factor, rel=1e-15)
    assert q["sigma_a_C0"].averaging.method == "normalized_maxwell_flux"
    assert q["sigma_a_C0"].averaging.energy_bounds_ev == (1e-5,10.)
    np.testing.assert_allclose(resolve_reference_rates(d),
        1.6*Avogadro/12.0107*1e-24*factor*np.array([.003862,5.575]), rtol=2e-15)
    assert d.composition == (("C0",1.),)
    assert all("JENDL-5" not in q.origin.source_library for q in d.quantities if q.unit == "barn")
    assert d.quality.verification_status == "unverified"  # Caller density is not upgraded.


def test_point_control_and_thermal_group_do_not_alias_each_other():
    from material import REFERENCE_CONTROL_PROFILE, REFERENCE_MAXWELL_PROFILE
    point = graphite_profile(REFERENCE_CONTROL_PROFILE)
    group = graphite_profile(REFERENCE_MAXWELL_PROFILE)
    p = {q.name:q for q in point.quantities}
    g = {q.name:q for q in group.quantities}
    assert p["sigma_s_C0"].values == 4.942
    assert p["sigma_a_C0"].values == .003860
    assert p["sigma_s_C0"].averaging.method == "point_evaluation"
    assert p["sigma_s_C0"].averaging.energy_bounds_ev == (.0253,.0253)
    assert point.material_id != group.material_id
    assert p["sigma_s_C0"].values != g["sigma_s_C0"].values


@pytest.mark.parametrize("profile_id", ["jendl5_point_0p0253ev", "jendl40_point_0p0253ev", "jendl40_maxwell_flux_0p0253ev"])
def test_all_reference_profile_material_archives_roundtrip(tmp_path, profile_id):
    from material import export_material_reference, load_material_reference, material_to_dict, resolve_reference_profile
    d = graphite_profile(profile_id)
    digest = export_material_reference(d,tmp_path / "profile.json")
    restored = load_material_reference(tmp_path / "profile.json",expected_sha256=digest)
    assert material_to_dict(restored) == material_to_dict(d)
    assert resolve_reference_profile([restored])["profile_id"] == profile_id


@pytest.mark.parametrize("profile_id", ["epithermal", "fast", "", None, True])
def test_unknown_or_invalid_profile_never_falls_back(profile_id):
    from material import get_reference_transport_dataset, REFERENCE_APPROXIMATION
    with pytest.raises((ValueError,TypeError)):
        get_reference_transport_dataset("Вода (H₂O)", approximation=REFERENCE_APPROXIMATION, profile_id=profile_id)


@pytest.mark.parametrize("profile_id", ["jendl40_point_0p0253ev", "jendl40_maxwell_flux_0p0253ev"])
def test_incomplete_water_profile_is_unavailable(profile_id):
    from material import get_reference_transport_dataset, get_reference_profiles
    with pytest.raises(ValueError,match="unavailable.*Complete source-checked"):
        get_reference_transport_dataset("Вода (H₂O)",profile_id=profile_id,
            approximation=get_reference_profiles()[profile_id]["approximation"])


def test_maxwell_group_requires_its_own_explicit_approximation():
    from material import get_reference_transport_dataset, REFERENCE_MAXWELL_PROFILE, REFERENCE_APPROXIMATION
    with pytest.raises(ValueError,match="free_gas_fixed_spectrum"):
        get_reference_transport_dataset("Графит",profile_id=REFERENCE_MAXWELL_PROFILE,approximation=REFERENCE_APPROXIMATION)


@pytest.mark.parametrize("change", ["bounds", "weight", "source", "raw_macs", "group_value", "isotopes", "profile_id"])
def test_group_material_semantic_forgery_rejected(change):
    from material import REFERENCE_MAXWELL_PROFILE, resolve_reference_rates
    d = graphite_profile(REFERENCE_MAXWELL_PROFILE)
    if change in ("bounds","weight"):
        d = replace(d, quantities=tuple(replace(q,averaging=replace(q.averaging,
            **({"energy_bounds_ev":(1e-5,1.)} if change == "bounds" else {"weighting":"uniform spectrum"})))
            if q.averaging else q for q in d.quantities))
    if change == "source": d = replace(d,origin=replace(d.origin,source_url="https://example.org/fake"))
    if change in ("raw_macs","group_value"):
        name = "published_macs_sigma_s_C0" if change == "raw_macs" else "sigma_s_C0"
        d = replace(d,quantities=tuple(replace(q,values=1.) if q.name == name else q for q in d.quantities))
    if change == "isotopes": d = replace(d,composition=(("C12",1.),))
    if change == "profile_id": d = replace(d,material_id=d.material_id.replace("maxwell_flux","unknown"))
    with pytest.raises(ValueError,match="reference transport"):
        resolve_reference_rates(d)
