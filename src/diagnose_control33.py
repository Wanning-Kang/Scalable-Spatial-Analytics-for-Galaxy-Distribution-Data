"""Independently enumerate short pairs for the unusual original uniform control.

Uses query_pairs/sparse-distance edge enumeration plus explicit minimum-image
norms, rather than count_neighbors. Does not replace or alter original output.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
import run_analysis as ra

ROOT = Path(__file__).resolve().parents[1]


def histogram(a, b, pairs, edges):
    delta = np.abs(a[pairs[:, 0]]-b[pairs[:, 1]])
    distances = np.linalg.norm(np.minimum(delta, 75.-delta), axis=1)
    use = (distances > edges[0]) & (distances <= edges[-1])
    idx = np.searchsorted(edges, distances[use], side="left")-1
    return np.bincount(idx, minlength=len(edges)-1)


def main():
    d = ROOT/"results/science/snapshot_033_poisson_control"
    with np.load(d/"points.npz") as f:
        x = f["coords"]
    with np.load(d/"random_catalog.npz") as f:
        random = f["coords"]
    with np.load(d/"global.npz") as f:
        edges = f["edges"]
        saved = {k: f[k] for k in ["DD", "DR", "RR"]}
    # Enumerate only distances <=1 cMpc/h, enough for bins wholly inside [.2,1].
    first, second = cKDTree(x, boxsize=75), cKDTree(random, boxsize=75)
    dd = first.query_pairs(1., output_type="ndarray")
    rr = second.query_pairs(1., output_type="ndarray")
    cross = first.sparse_distance_matrix(second, 1., output_type="coo_matrix")
    dr = np.column_stack((cross.row, cross.col))
    actual = {"DD": histogram(x, x, dd, edges), "DR": histogram(x, random, dr, edges),
              "RR": histogram(random, random, rr, edges)}
    idx = np.flatnonzero((edges[:-1] >= .2) & (edges[1:] <= 1.))
    for key in saved:
        np.testing.assert_array_equal(actual[key][idx], saved[key][idx])
    n, nr = len(x), len(random)
    probability = 4*np.pi/3*(edges[1:]**3-edges[:-1]**3)/75**3
    ra.save_json(ROOT/"results/uniform_null_ensemble/original_033_pair_check.json", {
        "status": "three_small_scale_histograms_exactly_match_independent_edge_enumeration",
        "validated_bin_indices": idx, "method": "query_pairs/sparse edge enumeration followed by explicit minimum-image Euclidean norms",
        "original_saved_counts": {k: v[idx] for k, v in saved.items()},
        "independently_enumerated_counts": {k: v[idx] for k, v in actual.items()},
        "analytic_uniform_expected_counts": {"DD": (n*(n-1)/2*probability)[idx],
            "DR": (n*nr*probability)[idx], "RR": (nr*(nr-1)/2*probability)[idx]},
        "warning": "Confirms pair-count arithmetic for these bins; does not make the unusual null realisation pass a zero-signal test or validate the JK error model"})
    print("Original control33 DD/DR/RR match explicit periodic pair enumeration in all three small-scale bins")


if __name__ == "__main__":
    main()
