from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from feature_audit import audit_catalog, build_features


def catalogue():
    return {"coords": np.array([[0., 1., 1.], [9.5, 8., 8.], [4., 4., 4.]]),
            "subfind_id": np.array([5, 9, 15]), "snapshot": 99, "redshift": 0.,
            "boxsize_cMpc_h": 10., "coordinate_units": "cMpc/h",
            "stellar_mass_msun": np.array([1e9, 1e10, 1e11]),
            "stellar_particle_count": np.array([100, 200, 300]), "halo_id": np.array([0, 1, 1]),
            "data_kind": "TEST_ONLY"}


CFG = {"jackknife_grid": 2, "eda_boundary_width_cMpc_h": 1., "knn_k": 20, "knn_sensitivity_k": 10}


def features(cat=None, rho=None):
    return build_features(cat or catalogue(), np.array([1., 2., 3.]),
                          np.array([2., 4., 8.]) if rho is None else rho, np.array([2., 3., 4.]),
                          np.array(["low", "medium", "high"]), np.array([-1, 0, 0]),
                          np.array([0., .7, 1.]), np.array([.8, .2, 0.]), 2, CFG)


def test_eda_counts_units_and_periodic_seam():
    report = audit_catalog(catalogue(), CFG)
    assert report["n"] == sum(report["octant_counts"]) == sum(report["spatial_region_counts"]) == 3
    assert report["boundary_proximity"]["n"] == 2
    assert report["global_number_density"] == 3/1000
    assert report["log10_stellar_mass_msun"]["quantiles"]["P50"] == 10.


def test_features_preserve_identity_geometry_and_distinct_normalizations():
    output = features()
    np.testing.assert_array_equal(output["coords"], catalogue()["coords"])
    np.testing.assert_array_equal(output["object_key"], ["99:5", "99:9", "99:15"])
    np.testing.assert_allclose(output["rho_over_median"], [.5, 1., 2.])
    np.testing.assert_allclose(output["number_overdensity"], np.array([2., 4., 8.])/(3/1000)-1)
    assert output["valid_derived_features"].all()
    assert output["is_noise"].tolist() == [True, False, False]


def test_derived_invalids_retained_and_flagged_without_id_loss():
    output = features(rho=np.array([np.inf, 4., 8.]))
    assert output["valid_derived_features"].tolist() == [False, True, True]
    assert np.isnan(output["rho_over_median"][0])
    assert len(output["subfind_id"]) == 3


def test_snapshot_keys_do_not_claim_persistent_ids():
    second = catalogue()
    second["snapshot"] = 50
    assert not np.isin(features()["object_key"], features(second)["object_key"]).any()


@pytest.mark.parametrize("change", ["duplicate_ids", "out_of_box", "missing_units", "raw_units"])
def test_invalid_catalogue_not_silently_repaired(change):
    cat = catalogue()
    if change == "duplicate_ids":
        cat["subfind_id"][1] = 5
    elif change == "out_of_box":
        cat["coords"][0, 0] = 1000.
    elif change == "missing_units":
        cat.pop("coordinate_units")
    else:
        cat["coordinate_units"] = "ckpc/h"
    with pytest.raises(ValueError):
        audit_catalog(cat, CFG)
