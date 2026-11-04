"""Periodic spatial statistics used by the Galaxy experiment.

Distances use the minimum image of the original torus. Pair histograms are
unique unordered self-pairs in (edges[i], edges[i + 1]], excluding distance
zero. Local-coordinate selections are descriptive, not causal experiments.
"""

from __future__ import annotations

import math
import hashlib
import warnings

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _points(x, boxsize):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n, 3).")
    if not np.isfinite(x).all():
        raise ValueError("Coordinates must be finite.")
    boxsize = float(boxsize)
    if not np.isfinite(boxsize) or boxsize <= 0:
        raise ValueError("boxsize must be a finite positive scalar.")
    return np.ascontiguousarray(np.mod(x, boxsize))


def _edges(edges):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("edges must be finite, nonnegative, strictly increasing.")
    return edges


def pair_counts(x, y, edges, boxsize):
    """Return raw integer counts in (left, right] using periodic cKDTree.

    y=None means unique self-pairs. Positive-distance bins drop the diagonal
    and divide ordered self-pairs by two. Zero-distance duplicate points are
    excluded consistently with the documented histogram definition.
    """
    edges = _edges(edges)
    x = _points(x, boxsize)
    y = None if y is None else _points(y, boxsize)
    if len(x) == 0 or (y is not None and len(y) == 0):
        return np.zeros(len(edges) - 1, dtype=np.int64)
    tree_x = cKDTree(x, boxsize=boxsize)
    tree_y = tree_x if y is None else cKDTree(y, boxsize=boxsize)
    counts = np.asarray(tree_x.count_neighbors(
        tree_y, edges, cumulative=False), dtype=np.int64)[1:]
    if y is None:
        if np.any(counts % 2):
            raise AssertionError("Ordered positive self-counts must be even.")
        counts //= 2
    return counts


def brute_counts(x, y, edges, boxsize, block_size=256):
    """Exact blocked minimum-image reference with bounded temporary memory.

    Runtime remains quadratic, but memory is O(block_size**2), not O(n**2).
    This is a correctness baseline and a fair blockwise memory comparator.
    """
    edges = _edges(edges)
    x = _points(x, boxsize)
    is_self = y is None
    y = x if is_self else _points(y, boxsize)
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    out = np.zeros(len(edges) - 1, dtype=np.int64)
    for i in range(0, len(x), block_size):
        a = x[i:i + block_size]
        for j in range(i if is_self else 0, len(y), block_size):
            b = y[j:j + block_size]
            delta = np.abs(a[:, None, :] - b[None, :, :])
            delta = np.minimum(delta, float(boxsize) - delta)
            distance = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
            if is_self and i == j:
                distance = distance[np.triu_indices(len(a), k=1)]
            else:
                distance = distance.ravel()
            valid = (distance > edges[0]) & (distance <= edges[-1])
            indices = np.searchsorted(edges, distance[valid], side="left") - 1
            out += np.bincount(indices, minlength=len(out)).astype(np.int64)
    return out


def _ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=np.isfinite(b) & (b > 0))
    return out


def estimators(dd_raw, dr_raw, rr_raw, nd, nr):
    """Five normalized estimators, following Kerscher et al. (2000), Eq. 2.

    Counts are unordered for DD/RR and all cross-pairs for DR. Unpopulated
    denominator bins return NaN, rather than fabricated zeros. Hewett is
    (dd-dr)/rr, with no additional minus one.
    """
    dd_raw, dr_raw, rr_raw = np.broadcast_arrays(
        np.asarray(dd_raw, dtype=float), np.asarray(dr_raw, dtype=float),
        np.asarray(rr_raw, dtype=float))
    nd, nr = int(nd), int(nr)
    if nd < 0 or nr < 0 or any(np.any(a < 0) for a in (dd_raw, dr_raw, rr_raw)):
        raise ValueError("Sample sizes and counts must be nonnegative.")
    dd = _ratio(dd_raw, nd * (nd - 1) / 2)
    dr = _ratio(dr_raw, nd * nr)
    rr = _ratio(rr_raw, nr * (nr - 1) / 2)
    return {
        "natural": _ratio(dd, rr) - 1,
        "davis_peebles": _ratio(dd, dr) - 1,
        "hewett": _ratio(dd - dr, rr),
        "hamilton": _ratio(dd * rr, dr * dr) - 1,
        "landy_szalay": _ratio(dd - 2 * dr + rr, rr),
    }


def density_knn(x, boxsize, k):
    """Return kth-other-neighbor distance and k/V local number-density proxy.

    This is a descriptive galaxy number density, not stellar mass density.
    k/V has finite-k Poisson bias; preserve the convention across snapshots.
    """
    x = _points(x, boxsize)
    k = int(k)
    if not 1 <= k < len(x):
        raise ValueError("k must be between 1 and n-1.")
    distances, indices = cKDTree(x, boxsize=boxsize).query(x, k=k + 1, workers=1)
    distances[indices == np.arange(len(x))[:, None]] = np.inf
    dk = np.partition(distances, k - 1, axis=1)[:, k - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = k / ((4 / 3) * np.pi * dk ** 3)
    return dk, rho


def region_ids(x, L, grid):
    """Assign a fixed grid**3 periodic spatial-region ID to each object."""
    x = _points(x, L)
    grid = int(grid)
    if grid < 1:
        raise ValueError("grid must be positive.")
    cell = np.minimum(np.floor(x / L * grid).astype(int), grid - 1)
    return (cell[:, 0] * grid + cell[:, 1]) * grid + cell[:, 2]


def jackknife(x, random, edges, L, grid, random_cache=None):
    """Ordinary delete-one spatial jackknife using matched D/R masks.

    The original periodic L is retained: each replicate has a masked hole,
    not a smaller periodic box. This estimates internal spatial uncertainty;
    two-point cross-region pairs can bias conventional jackknife covariance.
    It does not include fluctuations of modes larger than the simulation.
    """
    x, random, edges = _points(x, L), _points(random, L), _edges(edges)
    grid = int(grid)
    if grid < 2:
        raise ValueError("Jackknife requires grid >= 2.")
    rd, rr = region_ids(x, L, grid), region_ids(random, L, grid)
    count = grid ** 3
    signature = hashlib.sha256()
    for values in (random, edges, np.asarray([L, grid], dtype=np.float64)):
        signature.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    random_signature = signature.hexdigest()
    if random_cache is not None:
        if (random_cache.get("random_signature") != random_signature
                or np.asarray(random_cache.get("rr")).shape != (count, len(edges)-1)
                or np.asarray(random_cache.get("n_random")).shape != (count,)):
            raise ValueError("Jackknife RR cache must match the exact random catalogue, bins, box and region grid.")
    reps, dd_list, dr_list, rr_list, nd_list, nr_list = [], [], [], [], [], []
    for region in range(count):
        xd, xr = x[rd != region], random[rr != region]
        dd = pair_counts(xd, None, edges, L)
        dr = pair_counts(xd, xr, edges, L)
        rrc = (pair_counts(xr, None, edges, L) if random_cache is None
               else np.asarray(random_cache["rr"])[region])
        if random_cache is not None and int(random_cache["n_random"][region]) != len(xr):
            raise ValueError("Jackknife RR cache has inconsistent random region counts.")
        reps.append(estimators(dd, dr, rrc, len(xd), len(xr))["landy_szalay"])
        dd_list.append(dd)
        dr_list.append(dr)
        rr_list.append(rrc)
        nd_list.append(len(xd))
        nr_list.append(len(xr))
    reps = np.asarray(reps)
    covariance = np.full((len(edges) - 1, len(edges) - 1), np.nan)
    valid = np.isfinite(reps).all(axis=0)
    if valid.any():
        centered = reps[:, valid] - reps[:, valid].mean(axis=0)
        covariance[np.ix_(valid, valid)] = (count - 1) / count * centered.T @ centered
    return {
        "replicates": reps, "covariance": covariance,
        "se": np.sqrt(np.diag(covariance)), "n_regions": count,
        "region_ids_data": rd, "region_ids_random": rr,
        "n_data": np.asarray(nd_list), "n_random": np.asarray(nr_list),
        "dd": np.asarray(dd_list), "dr": np.asarray(dr_list), "rr": np.asarray(rr_list),
        "random_signature": random_signature,
    }


def _periodic_distances(x, L):
    delta = np.abs(x[:, None, :] - x[None, :, :])
    delta = np.minimum(delta, L - delta)
    return np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))


def _sparse_periodic_graph(tree, x, graph_k):
    distances, indices = tree.query(x, k=graph_k + 1, workers=1)
    rows = np.broadcast_to(np.arange(len(x))[:, None], indices.shape)
    use = (indices != rows) & np.isfinite(distances) & (distances > 0)
    graph = coo_matrix((distances[use], (rows[use], indices[use])),
                       shape=(len(x), len(x))).tocsr()
    # maximum unions directed neighbor edges without summing reciprocal distances.
    return graph.maximum(graph.T)


def periodic_hdbscan(x, L, min_cluster_size, min_samples, graph_k=32,
                     max_graph_k=256, dense_limit=1500):
    """HDBSCAN on exact dense or approximated sparse periodic distances.

    Sparse mode omits distant graph edges and is not guaranteed to reproduce
    the complete-metric MST. Validate graph-k convergence and dense small-N
    agreement. No arbitrary long-distance bridges or max_dist are inserted.
    Contrib hdbscan min_samples counts other neighbors; sklearn differs by 1.
    Membership probabilities are strengths, not calibrated physical chances.
    """
    import hdbscan

    x = _points(x, L)
    min_cluster_size, min_samples = int(min_cluster_size), int(min_samples)
    if len(x) < 3 or not 2 <= min_cluster_size <= len(x) or not 1 <= min_samples < len(x):
        raise ValueError("Invalid HDBSCAN sample or cluster sizes.")
    settings = dict(min_cluster_size=min_cluster_size, min_samples=min_samples,
                    metric="precomputed", algorithm="generic", allow_single_cluster=False,
                    cluster_selection_method="eom", approx_min_span_tree=False)
    if len(x) <= int(dense_limit):
        distance = _periodic_distances(x, L)
        metadata = {"backend": "dense_periodic_exact", "graph_k": None,
                    "n": len(x), "approximation": False}
    else:
        if len(np.unique(x, axis=0)) != len(x):
            raise ValueError("Sparse periodic HDBSCAN requires distinct coordinates; audit duplicates first.")
        maximum = min(int(max_graph_k), len(x) - 1)
        q = max(int(graph_k), min_samples)
        if q > maximum:
            raise ValueError("max_graph_k must be >= graph_k and min_samples.")
        tree = cKDTree(x, boxsize=L)
        attempts = []
        while True:
            # A conservative temporary-buffer estimate, not a measured RSS.
            # kNN queries + symmetrization can allocate several copies; fail
            # explicitly rather than exhaust a laptop on a very rich graph.
            estimated_bytes = len(x)*(q+1)*160
            if estimated_bytes > 1_500_000_000:
                raise MemoryError(f"Estimated sparse graph working buffers {estimated_bytes/1e9:.2f} GB exceed the 1.5 GB safety limit; reduce catalogue/graph size explicitly.")
            distance = _sparse_periodic_graph(tree, x, q)
            components = connected_components(distance, directed=False, return_labels=False)
            degree = np.diff(distance.indptr)
            attempts.append({"graph_k": q, "components": int(components),
                             "min_degree": int(degree.min())})
            if components == 1 and degree.min() >= min_samples:
                break
            if q == maximum:
                raise RuntimeError(f"Periodic kNN graph remains unsuitable at k={q}: "
                                   f"{components} components, minimum degree {degree.min()}. "
                                   "Increase max_graph_k or use an explicitly smaller dense sample.")
            q = min(maximum, max(q + 1, 2 * q))
        metadata = {"backend": "sparse_periodic_knn", "graph_k": q,
                    "n": len(x), "approximation": True, "attempts": attempts,
                    "edges_directed": int(distance.nnz), "min_degree": int(degree.min())}
    clusterer = hdbscan.HDBSCAN(**settings).fit(distance)
    return {"labels": clusterer.labels_.copy(),
            "probabilities": clusterer.probabilities_.copy(),
            "outlier": np.asarray(clusterer.outlier_scores_).copy(),
            "persistence": clusterer.cluster_persistence_.copy(),
            "metadata": metadata}


def validate_periodic_clustering(x, L, min_cluster_size, min_samples, graph_ks=(16, 32, 64)):
    """Compare sparse graph truncations with dense periodic HDBSCAN on small N."""
    from sklearn.metrics import adjusted_rand_score

    x = _points(x, L)
    dense = periodic_hdbscan(x, L, min_cluster_size, min_samples, dense_limit=len(x))
    records = []
    for k in graph_ks:
        k = min(max(int(k), int(min_samples)), len(x) - 1)
        sparse = periodic_hdbscan(x, L, min_cluster_size, min_samples,
                                 graph_k=k, max_graph_k=k, dense_limit=0)
        records.append({"graph_k": k,
                        "adjusted_rand_index": float(adjusted_rand_score(dense["labels"], sparse["labels"])),
                        "noise_agreement": float(np.mean((dense["labels"] == -1) == (sparse["labels"] == -1))),
                        "n_clusters_dense": int(len(np.unique(dense["labels"][dense["labels"] >= 0]))),
                        "n_clusters_sparse": int(len(np.unique(sparse["labels"][sparse["labels"] >= 0])))})
    return records


def decomposition(x, labels, edges, L):
    """Count four mutually exclusive DD categories without enumerating all pairs.

    Shares describe raw pair composition, not contributions to excess xi.
    Enrichment divides observed counts by a combinatorial random-label null
    that preserves positions, category sizes, and the original all-DD curve.
    """
    x, edges = _points(x, L), _edges(edges)
    labels = np.asarray(labels)
    if labels.shape != (len(x),) or np.any(labels < -1):
        raise ValueError("labels must have one nonnegative cluster ID or -1 per point.")
    members, noise = labels >= 0, labels == -1
    all_dd = pair_counts(x, None, edges, L)
    member_dd = pair_counts(x[members], None, edges, L)
    noise_dd = pair_counts(x[noise], None, edges, L)
    cross = pair_counts(x[members], x[noise], edges, L)
    within = np.zeros_like(all_dd)
    possible_within = 0
    for label in np.unique(labels[members]):
        group = x[labels == label]
        within += pair_counts(group, None, edges, L)
        possible_within += len(group) * (len(group) - 1) // 2
    raw = {"within": within, "between": member_dd - within,
           "cluster_noise": cross, "noise_noise": noise_dd}
    if np.any(sum(raw.values()) != all_dd) or np.any(raw["between"] < 0):
        raise AssertionError("Pair categories must partition all DD exactly.")
    nm, nn = int(members.sum()), int(noise.sum())
    possible = {"within": possible_within,
                "between": nm * (nm - 1) // 2 - possible_within,
                "cluster_noise": nm * nn, "noise_noise": nn * (nn - 1) // 2}
    total_possible = len(x) * (len(x) - 1) // 2
    expected = {key: all_dd * (value / total_possible) if total_possible else
                np.full(len(all_dd), np.nan) for key, value in possible.items()}
    return {"all": all_dd, "raw": raw, "possible_pairs": possible,
            "shares": {key: _ratio(value, all_dd) for key, value in raw.items()},
            "expected_random_labels": expected,
            "enrichment": {key: _ratio(raw[key], expected[key]) for key in raw}}


def powerlaw_bin_average(edges, r0, gamma):
    """Volume-weighted shell average of (r0/r)**gamma; require gamma < 3."""
    edges = _edges(edges)
    gamma, r0 = float(gamma), float(r0)
    if not 0 < gamma < 3 or r0 <= 0:
        raise ValueError("Power law requires r0 > 0 and 0 < gamma < 3.")
    low, high = edges[:-1], edges[1:]
    return (3 * r0 ** gamma / (3 - gamma)
            * (high ** (3 - gamma) - low ** (3 - gamma)) / (high ** 3 - low ** 3))


def fit_powerlaw(edges, xi, cov, fit_range):
    """Fit a shell-averaged power law with GLS and explicit covariance ridge.

    Only positive finite xi bins wholly inside fit_range are fitted. The
    covariance ridge is max(1e-6 * median positive diagonal, -lambda_min plus
    that floor), recorded in the result. Poor fits and bound hits are exposed;
    one power law need not describe all galaxy-clustering scales.
    """
    edges = _edges(edges)
    xi, cov = np.asarray(xi, dtype=float), np.asarray(cov, dtype=float)
    if xi.shape != (len(edges) - 1,) or cov.shape != (len(xi), len(xi)):
        raise ValueError("xi/cov dimensions must match edges.")
    lower, upper = map(float, fit_range)
    valid = ((edges[:-1] >= lower) & (edges[1:] <= upper)
             & np.isfinite(xi) & (xi > 0) & np.isfinite(np.diag(cov)))
    idx = np.flatnonzero(valid)
    if len(idx) < 4:
        return {"success": False, "message": "Fewer than four valid positive fitting bins.",
                "fit_indices": idx, "n_bins": len(idx)}
    c = cov[np.ix_(idx, idx)]
    if not np.isfinite(c).all():
        return {"success": False, "message": "Non-finite selected covariance.",
                "fit_indices": idx, "n_bins": len(idx)}
    c = (c + c.T) / 2
    diagonal = np.diag(c)
    scale = float(np.median(diagonal[diagonal > 0])) if np.any(diagonal > 0) else float(np.median(xi[idx] ** 2) * 1e-6)
    floor = max(scale * 1e-6, np.finfo(float).eps)
    ridge = max(floor, -float(np.linalg.eigvalsh(c).min()) + floor)
    c_regularized = c + ridge * np.eye(len(idx))
    chol = cholesky(c_regularized, lower=True)
    rcenter = np.sqrt(edges[:-1] * edges[1:])
    initial_r0 = float(np.median(rcenter[idx] * xi[idx] ** (1 / 1.8)))
    initial_r0 = np.clip(initial_r0, 1e-5, 1e5)

    def residual(theta):
        model = powerlaw_bin_average(edges, theta[0], theta[1])[idx]
        return solve_triangular(chol, model - xi[idx], lower=True)

    result = least_squares(residual, [initial_r0, 1.8], bounds=([1e-6, 0.05], [1e6, 2.9]))
    model = powerlaw_bin_average(edges, *result.x)
    difference = xi[idx] - model[idx]
    chi_square = float(np.sum(result.fun ** 2))
    try:
        parameter_cov = np.linalg.inv(result.jac.T @ result.jac)
        param_se = np.sqrt(np.diag(parameter_cov))
    except np.linalg.LinAlgError:
        parameter_cov = np.full((2, 2), np.nan)
        param_se = np.full(2, np.nan)
    bound_hit = bool(np.any(result.active_mask))
    return {"success": bool(result.success), "message": result.message,
            "r0": float(result.x[0]), "gamma": float(result.x[1]),
            "parameter_covariance": parameter_cov, "r0_se": float(param_se[0]),
            "gamma_se": float(param_se[1]), "predicted": model,
            "residual": xi - model, "fit_indices": idx, "n_bins": len(idx),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "chi_square": chi_square, "reduced_chi_square": chi_square / (len(idx) - 2),
            "covariance_ridge": ridge, "covariance_condition_number": float(np.linalg.cond(c_regularized)),
            "bound_hit": bound_hit, "fit_range": [lower, upper]}
