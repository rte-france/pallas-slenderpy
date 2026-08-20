import json

import numpy as np
import pytest

from slenderpy.future.simulation import Parameters, Results

# --- Parameters ------------------------------------------------------------


def test_parameters_derived_counts():
    p = Parameters(ns=11, t0=0.0, tf=1.0, dt=0.01, dr=0.1)
    assert p.nt == 100
    assert p.nr == 10
    assert p.rr == 10


def test_parameters_defaults():
    p = Parameters()
    assert p.ns == 11
    assert len(p.los) == 11
    assert all(0.0 < s < 1.0 for s in p.los)
    assert p.los == sorted(p.los)


def test_parameters_los_int_becomes_interior_points():
    p = Parameters(los=5)
    assert len(p.los) == 5
    assert all(0.0 < s < 1.0 for s in p.los)
    assert p.los == sorted(p.los)


def test_parameters_los_single_point():
    p = Parameters(los=1)
    assert p.los == [0.5]


def test_parameters_time_vectors():
    p = Parameters(ns=11, t0=0.0, tf=1.0, dt=0.01, dr=0.1)
    tv = p.time_vector()
    assert len(tv) == 1 + p.nt
    assert tv[0] == pytest.approx(0.0)
    assert tv[-1] == pytest.approx(1.0)
    tvo = p.time_vector_output()
    assert len(tvo) == 1 + p.nr
    assert tvo[0] == pytest.approx(0.0)
    assert tvo[-1] == pytest.approx(1.0)


def test_parameters_accepts_int_args():
    p = Parameters(ns=11, t0=0, tf=10, dt=1, dr=2)
    assert isinstance(p.t0, float)
    assert isinstance(p.tf, float)
    assert p.nt == 10
    assert p.nr == 5
    assert p.rr == 2


def test_parameters_int_tf_default_rest():
    p = Parameters(tf=1)
    assert p.tf == 1.0
    assert isinstance(p.tf, float)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ns": 11.0},  # ns not int
        {"ns": 5},  # ns < 6
        {"t0": 1.0, "tf": 1.0},  # tf <= t0
        {"dt": 0.0},  # dt == 0 (divide-by-zero guard)
        {"dt": -0.01},  # dt < 0
        {"dr": 0.001, "dt": 0.01},  # dr < dt
        {"los": [0.5, 1.5]},  # los >= 1
        {"los": [0.5, 0.2]},  # not ascending
        {"los": [0.5, 0.5]},  # duplicate
        {"pp": "yes"},  # wrong type
    ],
)
def test_parameters_invalid(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Parameters(**kwargs)


# --- Results ---------------------------------------------------------------


def _make_results():
    lot = [0.0, 1.0, 2.0]
    lov = ["scalar", "vector"]
    lov_dims = [1, 2]
    los = [0.25, 0.5, 0.75]
    return Results(lot=lot, lov=lov, lov_dims=lov_dims, los=los), lot, lov, los


def test_results_accessors():
    res, lot, lov, los = _make_results()
    assert res.lot() == pytest.approx(lot)
    assert res.los() == pytest.approx(los)
    assert set(res.lov()) == set(lov)


def test_results_update_scalar_and_vector():
    res, lot, lov, los = _make_results()
    res.update(0, None, ["scalar"], [3.14])
    assert float(res["scalar"][0].values) == pytest.approx(3.14)

    # vector: linear source 0..4 over s=[0,1], interpolated onto los.
    s = np.array([0.0, 1.0])
    values = np.array([0.0, 4.0])
    res.update(1, s, ["vector"], [values])
    assert res["vector"][1].values == pytest.approx([1.0, 2.0, 3.0])


def test_results_lov_dims_default_all_vector():
    res = Results(lot=[0.0, 1.0], lov=["a"], los=[0.5])
    assert res.lov_dims["a"] == 2
    res.update(0, np.array([0.0, 1.0]), ["a"], [np.array([2.0, 2.0])])
    assert res["a"][0].values == pytest.approx([2.0])


def test_results_invalid_lov_dim_raises():
    with pytest.raises(ValueError):
        Results(lot=[0.0, 1.0], lov=["a"], lov_dims=[3], los=[0.5])


def test_results_set_state():
    res, *_ = _make_results()
    res.set_state({"y": 1})
    assert res.state == {"y": 1}


def test_results_dump_load_roundtrip(tmp_path):
    res, lot, lov, los = _make_results()
    res.update(0, None, ["scalar"], [1.5])
    res.set_state({"k": 7})

    f = tmp_path / "res.pkl"
    res.dump(str(f))

    loaded = Results(filename=str(f))
    assert loaded.lot() == pytest.approx(lot)
    assert loaded.lov_dims == res.lov_dims
    assert loaded.state == {"k": 7}
    # The load fix must restore lov_dims so update() still works.
    loaded.update(1, None, ["scalar"], [2.5])
    assert float(loaded["scalar"][1].values) == pytest.approx(2.5)


def test_results_drop_variable_position_and_time():
    res, lot, lov, los = _make_results()
    s = np.array(los)
    for k in range(len(lot)):
        res.update(k, None, ["scalar"], [float(k)])
        res.update(k, s, ["vector"], [np.array([10.0, 20.0, 30.0])])

    res.drop(lov=["scalar"])
    assert "scalar" not in res.lov()
    assert "vector" in res.lov()

    res.drop(los=[0.5])
    assert 0.5 not in res.los()

    res.drop(tmin=1.0, tmax=1.0)
    assert res.lot() == pytest.approx([1.0])


def test_results_to_json_roundtrip():
    res, lot, lov, los = _make_results()
    s = np.array(los)
    for k in range(len(lot)):
        res.update(k, None, ["scalar"], [float(k + 1)])
        res.update(k, s, ["vector"], [np.array([k, k, k], dtype=float)])

    out = json.loads(res.to_json())
    assert out["time"] == pytest.approx(lot)
    assert out["curv"] == pytest.approx(los)
    assert out["scalar"] == pytest.approx([1.0, 2.0, 3.0])
    # vector is stored as a list over positions of time-series.
    assert len(out["vector"]) == len(los)
    assert out["vector"][0] == pytest.approx([0.0, 1.0, 2.0])
