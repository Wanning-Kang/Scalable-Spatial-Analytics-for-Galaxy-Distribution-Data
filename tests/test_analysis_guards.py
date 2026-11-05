"""Small, mocked control-flow tests; no complete experiment is executed."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import run_analysis as analysis


def fit_result(r0=3., gamma=1.7, indices=(2, 3, 4, 5), **kwargs):
    return {"success": True, "bound_hit": False, "fit_indices": np.array(indices),
            "r0": r0, "gamma": gamma, **kwargs}


def test_parameter_jk_same_interior_bins_has_correct_prefactor():
    full = fit_result()
    replicates = [fit_result(2., 1.5), fit_result(3., 1.7), fit_result(4., 1.9)]
    result = analysis.parameter_jackknife_summary(full, replicates)
    values = np.array([[2., 1.5], [3., 1.7], [4., 1.9]])
    d = values-values.mean(axis=0)
    assert result["status"] == "approximate_internal_spatial_uncertainty"
    np.testing.assert_allclose(result["covariance"], 2/3*d.T@d)
    np.testing.assert_allclose(result["approx_95_normal_intervals"][:, 0],
                               [3., 1.7]-1.96*np.sqrt(np.diag(2/3*d.T@d)))


@pytest.mark.parametrize("full,replicates,reason", [
    (fit_result(success=False), [fit_result(), fit_result()], "one_or_more_fits_failed"),
    (fit_result(), [fit_result(), fit_result(success=False)], "one_or_more_fits_failed"),
    (fit_result(bound_hit=True), [fit_result(), fit_result()], "one_or_more_fits_hit_parameter_bounds"),
    (fit_result(), [fit_result(), fit_result(bound_hit=True)], "one_or_more_fits_hit_parameter_bounds"),
    (fit_result(), [fit_result(), fit_result(indices=(2, 3, 4, 6))], "delete_region_fitting_bins_differ_from_full_sample"),
    (fit_result(), [fit_result()], "fewer_than_two_delete_region_fits"),
    (fit_result(), [fit_result(), fit_result(r0=np.nan)], "nonfinite_fitted_parameters"),
])
def test_parameter_jk_does_not_certify_incomparable_or_bound_fits(full, replicates, reason):
    result = analysis.parameter_jackknife_summary(full, replicates)
    assert result["status"] == "unavailable"
    assert result["reason"] == reason
    assert "approx_95_normal_intervals" not in result


def cluster_result(actual_k, labels=(0, 0, 1, -1), approximation=True):
    return {"labels": np.array(labels),
            "metadata": {"graph_k": actual_k, "approximation": approximation,
                         "backend": "sparse_periodic_knn" if approximation else "dense_periodic_exact"}}


def graph_cfg(cap=256):
    return {"graph_k": 32, "max_graph_k": cap, "min_cluster_size": 40, "min_samples": 10}


def test_higher_graph_starts_from_primary_actual_k_not_configured_k(monkeypatch):
    calls = []
    def fake(x, box, size, samples, **kwargs):
        calls.append(kwargs)
        return cluster_result(256)
    monkeypatch.setattr(analysis.sp, "periodic_hdbscan", fake)
    primary = cluster_result(128)
    result = analysis.graph_k_sensitivity(np.zeros((400, 3)), 75., graph_cfg(), primary)
    assert calls == [{"graph_k": 256, "max_graph_k": 256, "dense_limit": 0}]
    assert result["status"] == "available"
    assert result["primary"]["graph_k"] == 128
    assert result["higher"]["graph_k"] == 256
    assert result["ari_including_noise"] == 1.


@pytest.mark.parametrize("primary,n,reason", [
    (cluster_result(256), 400, "no_larger_graph_k_within_configured_limit"),
    (cluster_result(63), 64, "no_larger_graph_k_within_configured_limit"),
    (cluster_result(None, approximation=False), 20, "primary_uses_exact_dense_distances"),
])
def test_graph_limit_or_exact_primary_does_not_repeat_comparison(monkeypatch, primary, n, reason):
    def no_call(*args, **kwargs):
        pytest.fail("an unavailable comparison must not rerun an identical graph")
    monkeypatch.setattr(analysis.sp, "periodic_hdbscan", no_call)
    result = analysis.graph_k_sensitivity(np.zeros((n, 3)), 75., graph_cfg(), primary)
    assert result["status"] == "unavailable"
    assert result["reason"] == reason
    assert "ari_including_noise" not in result


@pytest.mark.parametrize("error", [MemoryError("safety guard"), RuntimeError("disconnected")])
def test_unavailable_larger_graph_is_explicit_not_fake_ari(monkeypatch, error):
    def fake(*args, **kwargs):
        raise error
    monkeypatch.setattr(analysis.sp, "periodic_hdbscan", fake)
    result = analysis.graph_k_sensitivity(np.zeros((400, 3)), 75., graph_cfg(), cluster_result(128))
    assert result["status"] == "unavailable"
    assert result["error_type"] == type(error).__name__
    assert result["requested_graph_k"] == 256
    assert "ari_including_noise" not in result


def test_unexpected_identical_actual_graph_does_not_get_ari(monkeypatch):
    monkeypatch.setattr(analysis.sp, "periodic_hdbscan", lambda *args, **kw: cluster_result(128))
    result = analysis.graph_k_sensitivity(np.zeros((400, 3)), 75., graph_cfg(), cluster_result(128))
    assert result["reason"] == "returned_graph_k_is_not_larger_than_primary"
    assert result["higher"]["graph_k"] == 128
    assert "ari_including_noise" not in result


def test_subset_jackknife_receives_full_sample_random_cache(monkeypatch, tmp_path):
    cache = {"random_signature": "same exact R / bins / L / grid"}
    random = np.zeros((6, 3))
    seen = []
    def fake_jk(x, r, edges, box, grid, random_cache=None):
        assert r is random
        seen.append(random_cache)
        return {"replicates": np.ones((2, 4)), "covariance": np.eye(4)}
    monkeypatch.setattr(analysis.sp, "jackknife", fake_jk)
    monkeypatch.setattr(analysis.sp, "fit_powerlaw", lambda *args: fit_result())
    monkeypatch.setattr(analysis, "ls_sample", lambda *args: {
        "DD": np.ones(4), "DR": np.ones(4), "RR": np.ones(4), "xi": np.ones(4)})
    full = {"RR": np.ones(4), "xi": np.ones(4),
            "jk_replicates": np.ones((2, 4)), "jk_cache": cache}
    analysis.subset_analysis(np.zeros((4, 3)), {"test": np.array([True, True, False, False])},
                             random, np.arange(5), 75.,
                             {"jackknife_grid": 2, "fit_range_cMpc_h": [.2, 3.]}, full, tmp_path)
    assert len(seen) == 1 and seen[0] is cache
