"""Focused tests for the versioned siim2d pickle envelope."""
import importlib
import pickle

import numpy as np
import pytest
import xarray as xr

from siim import __version__


siim2d_module = importlib.import_module("siim.siim2d")
siim2d = siim2d_module.siim


def _model_for_save():
    """A minimal object is enough to exercise envelope serialization."""
    model = object.__new__(siim2d)
    model._user_params = {}
    model.ds_out = xr.Dataset()
    return model


def _valid_payload():
    return {
        "save_format": "siim.model-state.pickle",
        "schema_version": 1,
        "siim_version": __version__,
        "model_identity": "siim.siim2d.siim",
        "_user_params": {},
        "ds_out": xr.Dataset(),
    }


def _dump(path, payload):
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def test_save_writes_versioned_envelope(tmp_path):
    path = _model_for_save().save(tmp_path / "state")

    assert path == tmp_path / "state.pkl"
    with path.open("rb") as stream:
        payload = pickle.load(stream)

    assert set(payload) == {
        "save_format", "schema_version", "siim_version", "model_identity",
        "_user_params", "ds_out",
    }
    assert payload["save_format"] == "siim.model-state.pickle"
    assert payload["schema_version"] == 1
    assert payload["siim_version"] == __version__
    assert payload["model_identity"] == "siim.siim2d.siim"
    assert not list(tmp_path.glob(".state.pkl.*.tmp"))


def test_versioned_save_load_reconstructs_outputs_without_running(tmp_path):
    params = {
        "U": 1e-3, "zELA": 150, "beta": 1e-2, "P": 2,
        "alpha_g": 12, "Ko": 2e-6, "n": 1, "ce": 1e-4, "nu": 2,
        "sliding_law": "power", "lambda_p": 500, "k": 0.9,
        "T": 1.0, "nt": 2, "nt_out": 1, "Lx": 2e4, "Ly": 2e4,
        "nx": 3, "ny": 3, "seed": 1, "progress_bar": False,
    }
    shape = (1, 3, 3)
    floats = np.arange(9, dtype=float).reshape(shape)
    receivers = np.arange(9, dtype=np.int32).reshape(shape)
    data_vars = {
        name: (("time", "y", "x"), floats.copy())
        for name in (
            "topography__elevation", "glacial_flow__area",
            "glacial_flow__ice_flux", "glacial_flow__water_flux",
            "glacial_spl__erosion_rate", "glacial_spl__denudation",
        )
    }
    data_vars["glacial_spl__ice_thickness"] = (
        ("time", "y", "x"), np.ones(shape))
    data_vars["glacial_flow__receivers_2d"] = (
        ("time", "y", "x"), receivers)
    data_vars["glacial_flow__basin_ids"] = (
        ("time", "y", "x"), np.zeros(shape, dtype=np.int32))
    data_vars["glacial_flow__stack_2d"] = (
        ("time", "y", "x"), receivers)
    dataset = xr.Dataset(
        data_vars,
        coords={"time": [0.0], "x": [0.0, 1e4, 2e4],
                "y": [0.0, 1e4, 2e4], "tstep": ("time", [0.0])},
    )

    model = siim2d(params)
    model.ds_out = dataset
    loaded = siim2d.load(model.save(tmp_path / "roundtrip.pkl"))

    assert loaded._user_params == params
    assert np.array_equal(loaded.H_out, np.ones(shape))
    assert np.array_equal(loaded.zb_out, floats)
    assert np.array_equal(loaded.z_out, floats + 1.5)
    assert np.array_equal(loaded.output_times, np.array([0.0]))


def test_failed_save_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "state.pkl"
    original = b"existing saved model"
    path.write_bytes(original)

    def fail_after_partial_write(_state, stream, protocol=None):
        stream.write(b"partial replacement")
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(siim2d_module.pickle, "dump", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated serialization failure"):
        _model_for_save().save(path)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".state.pkl.*.tmp"))


def test_load_rejects_unversioned_legacy_payload(tmp_path):
    path = tmp_path / "legacy.pkl"
    _dump(path, {"_user_params": {}, "ds_out": xr.Dataset()})

    with pytest.raises(ValueError, match="Unsupported legacy SIIM saved model"):
        siim2d.load(path)


def test_load_rejects_non_mapping_payload(tmp_path):
    path = tmp_path / "not-a-mapping.pkl"
    _dump(path, ["not", "a", "model"])

    with pytest.raises(ValueError, match="top-level pickle payload must be a dict"):
        siim2d.load(path)


def test_load_rejects_missing_required_key(tmp_path):
    path = tmp_path / "missing-key.pkl"
    payload = _valid_payload()
    del payload["siim_version"]
    _dump(path, payload)

    with pytest.raises(ValueError, match=r"missing required key.*siim_version"):
        siim2d.load(path)


@pytest.mark.parametrize(("changes", "match"), [
    ({"save_format": 1}, "'save_format' must be a string"),
    ({"save_format": "another.format"}, "Unsupported SIIM save format"),
    ({"schema_version": True}, "'schema_version' must be an integer"),
    ({"schema_version": 99}, "Unsupported SIIM save schema version 99"),
    ({"siim_version": ""}, "'siim_version' must be a non-empty string"),
    ({"model_identity": 1}, "'model_identity' must be a string"),
    ({"model_identity": "siim.siim1d.siim"}, "load it with the matching model class"),
    ({"_user_params": []}, "'_user_params' must be a dict"),
    ({"ds_out": {}}, "'ds_out' must be an xarray.Dataset"),
])
def test_load_rejects_invalid_envelope_values(tmp_path, changes, match):
    path = tmp_path / "invalid-envelope.pkl"
    payload = _valid_payload()
    payload.update(changes)
    _dump(path, payload)

    with pytest.raises(ValueError, match=match):
        siim2d.load(path)


def test_load_reports_invalid_pickle(tmp_path):
    path = tmp_path / "corrupt.pkl"
    path.write_bytes(b"this is not a pickle")

    with pytest.raises(ValueError, match="pickle is invalid or incompatible"):
        siim2d.load(path)
