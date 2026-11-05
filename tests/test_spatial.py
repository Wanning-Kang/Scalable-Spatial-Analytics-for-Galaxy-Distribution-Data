"""Scientific invariants: periodic metric, endpoint rules, masks and labels."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial import (pair_counts, brute_counts, estimators, density_knn,
                     region_ids, jackknife, decomposition, periodic_hdbscan,
                     fit_powerlaw, powerlaw_bin_average)


def line(values):
    return np.column_stack([values, np.zeros((len(values), 2))])


def test_periodic_self_endpoints_and_unique_pairs():
    x = line([0, 0.25, 0.75, 0.5])
    edges = np.array([0, 0.25, 0.5])
    assert np.array_equal(pair_counts(x, None, edges, 1), [4, 2])
    assert np.array_equal(brute_counts(x, None, edges, 1, block_size=2), [4, 2])


def test_zero_distance_exclusion_and_cross_endpoints():
    assert np.array_equal(pair_counts(line([0, 0, 0.25]), None, [0, 0.25, 0.5], 1), [2, 0])
    x, y = line([0, 0.25, 0.75, 0.5]), line([0.125, 0.875])
    edges = [0, 0.125, 0.375, 0.5]
    assert np.array_equal(pair_counts(x, y, edges, 1), [4, 4, 0])
    assert np.array_equal(brute_counts(x, y, edges, 1, block_size=2), [4, 4, 0])


def test_random_blocked_reference_matches_tree():
    rng = np.random.default_rng(12)
    x, y = rng.uniform(0, 1, (73, 3)), rng.uniform(0, 1, (91, 3))
    edges = np.linspace(0, 0.7, 12)
    for other in (None, y):
        assert np.array_equal(pair_counts(x, other, edges, 1),
                              brute_counts(x, other, edges, 1, block_size=13))


def test_five_estimators_normalization_and_empty_bins():
    results = estimators([2, 4], [10, 20], [9, 18], 5, 10)
    for name, value in results.items():
        np.testing.assert_allclose(value, 0, atol=1e-15, err_msg=name)
    xi = estimators([3], [10], [9], 5, 10)
    np.testing.assert_allclose(xi["hewett"], [0.5])
    assert all(np.isnan(a).all() for a in estimators([0], [0], [0], 0, 0).values())


def test_knn_excludes_self_and_wraps():
    dk, rho = density_knn(line([0, 0.25, 0.75, 0.5]), 1, 1)
    np.testing.assert_allclose(dk, 0.25)
    np.testing.assert_allclose(rho, 1 / ((4 / 3) * np.pi * 0.25 ** 3))


def test_regions_and_matched_jackknife_masks():
    x = np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.9, 0.9]])
    np.testing.assert_array_equal(region_ids(x, 1, 2), [0, 4, 3])
    rng = np.random.default_rng(34)
    data, random = rng.uniform(0, 1, (80, 3)), rng.uniform(0, 1, (160, 3))
    edges = np.linspace(0.1, 0.6, 7)
    result = jackknife(data, random, edges, 1, 2)
    assert result["replicates"].shape == (8, 6)
    assert result["covariance"].shape == (6, 6)
    for region in (0, 7):
        xd = data[region_ids(data, 1, 2) != region]
        xr = random[region_ids(random, 1, 2) != region]
        dd, dr, rr = pair_counts(xd, None, edges, 1), pair_counts(xd, xr, edges, 1), pair_counts(xr, None, edges, 1)
        np.testing.assert_array_equal(result["dd"][region], dd)
        np.testing.assert_array_equal(result["dr"][region], dr)
        np.testing.assert_array_equal(result["rr"][region], rr)
        np.testing.assert_allclose(result["replicates"][region],
                                   estimators(dd, dr, rr, len(xd), len(xr))["landy_szalay"])
    centered = result["replicates"] - result["replicates"].mean(axis=0)
    np.testing.assert_allclose(result["covariance"], 7 / 8 * centered.T @ centered)


def test_decomposition_partition_and_random_label_expectation():
    x = line([0, 0.125, 0.25, 0.5, 0.75, 0.875])
    labels = np.array([0, 0, 1, 1, -1, -1])
    result = decomposition(x, labels, [0, 0.25, 0.5], 1)
    np.testing.assert_array_equal(sum(result["raw"].values()), result["all"])
    assert result["possible_pairs"] == {"within": 2, "between": 4, "cluster_noise": 8, "noise_noise": 1}
    np.testing.assert_allclose(sum(result["expected_random_labels"].values()), result["all"])
    np.testing.assert_allclose(sum(result["shares"].values()), 1)


def test_jackknife_random_cache_preserves_counts_and_rejects_other_geometry():
    rng = np.random.default_rng(124)
    data, random = rng.uniform(0, 1, (60, 3)), rng.uniform(0, 1, (100, 3))
    edges = np.linspace(.05, .4, 5)
    full = jackknife(data, random, edges, 1, 2)
    direct = jackknife(data[:40], random, edges, 1, 2)
    cached = jackknife(data[:40], random, edges, 1, 2, random_cache=full)
    for key in ["dd", "dr", "rr", "replicates", "covariance"]:
        np.testing.assert_allclose(direct[key], cached[key], equal_nan=True)
    with pytest.raises(ValueError, match="exact random"):
        jackknife(data[:40], random + .001, edges, 1, 2, random_cache=full)


def test_periodic_hdbscan_connects_a_face_spanning_cluster():
    rng = np.random.default_rng(56)
    first = (np.array([0, 0.3, 0.3]) + rng.normal(0, 0.012, (18, 3))) % 1
    second = np.array([0.5, 0.7, 0.7]) + rng.normal(0, 0.012, (18, 3))
    x = np.vstack([first, second])
    result = periodic_hdbscan(x, 1, 8, 4, dense_limit=100)
    assert len(np.unique(result["labels"][:18])) == 1
    assert result["labels"][0] >= 0
    assert result["labels"][0] != result["labels"][18]
    assert result["metadata"]["backend"] == "dense_periodic_exact"
    # Full neighbor graph must reproduce exact dense periodic clustering.
    sparse = periodic_hdbscan(x, 1, 8, 4, graph_k=len(x)-1,
                              max_graph_k=len(x)-1, dense_limit=0)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(result["labels"], sparse["labels"]) == 1


def test_powerlaw_shell_average_fit_and_explicit_failure():
    edges = np.geomspace(0.2, 8, 13)
    truth = powerlaw_bin_average(edges, 3.2, 1.7)
    cov = np.diag((truth * 0.03) ** 2)
    fit = fit_powerlaw(edges, truth, cov, (0.2, 8))
    assert fit["success"]
    np.testing.assert_allclose([fit["r0"], fit["gamma"]], [3.2, 1.7], rtol=1e-5)
    assert fit["covariance_ridge"] > 0
    assert not fit["bound_hit"]
    assert not fit_powerlaw(edges, -truth, cov, (0.2, 8))["success"]
