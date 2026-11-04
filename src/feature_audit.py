"""Explicit EDA and per-object feature engineering; preserve physical geometry.

Coordinates stay in cMpc/h, not z-scored. Cluster IDs are categorical. Subfind
IDs are only meaningful within a snapshot, so joins use (snapshot, Subfind ID).
No visualization or claim about real TNG outcomes is produced here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"n": int(values.size), "finite_n": int(finite.size),
            "invalid_n": int(values.size-finite.size),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "quantiles": dict(zip(["P05", "P25", "P50", "P75", "P95"],
                                   np.quantile(finite, [.05, .25, .5, .75, .95]).tolist())) if finite.size else {}}


def validate_catalog(cat):
    x = np.asarray(cat["coords"], dtype=float)
    box = float(cat["boxsize_cMpc_h"])
    ids = np.asarray(cat["subfind_id"])
    if x.ndim != 2 or x.shape[1] != 3 or not len(x):
        raise ValueError("coords must be a nonempty (N,3) catalogue")
    if not np.isfinite(box) or box <= 0 or not np.isfinite(x).all() or np.any(x < 0) or np.any(x >= box):
        raise ValueError("finite coordinates must be in [0,L); no silent unit repair or wrapping")
    if str(np.asarray(cat.get("coordinate_units", ""))) != "cMpc/h":
        raise ValueError("explicit coordinate_units=cMpc/h is required")
    if ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("subfind_id must be a nonnegative integer vector matching coords")
    if np.unique(ids).size != len(ids):
        raise ValueError("duplicate Subfind IDs within snapshot; cannot join features safely")
    for name in ("stellar_mass_msun", "stellar_particle_count", "halo_id"):
        if np.asarray(cat[name]).shape != (len(x),):
            raise ValueError(f"{name} must match catalogue rows")
    return x, box, ids


def audit_catalog(cat, cfg):
    x, box, ids = validate_catalog(cat)
    width = float(cfg.get("eda_boundary_width_cMpc_h", 1.))
    grid = int(cfg["jackknife_grid"])
    if not np.isfinite(width) or not 0 < width < box/2 or grid < 1:
        raise ValueError("EDA boundary width must be in (0,L/2), grid must be positive")
    mass = np.asarray(cat["stellar_mass_msun"], dtype=float)
    logmass = np.full(len(x), np.nan)
    good = np.isfinite(mass) & (mass > 0)
    logmass[good] = np.log10(mass[good])
    cells = np.minimum((x/box*grid).astype(int), grid-1)
    regions = cells[:, 0]*grid**2 + cells[:, 1]*grid + cells[:, 2]
    octant = (x >= box/2).astype(int) @ np.array([4, 2, 1])
    excess = len(x)-len(np.unique(x, axis=0))
    return {"data_kind": str(cat.get("data_kind", "UNSPECIFIED")),
            "snapshot": int(cat["snapshot"]), "n": len(x), "coordinate_units": "cMpc/h",
            "boxsize_cMpc_h": box, "coordinate_min": x.min(axis=0).tolist(),
            "coordinate_max": x.max(axis=0).tolist(), "unique_id_n": len(ids),
            "duplicate_coordinate_excess": excess,
            "duplicate_coordinate_policy": "retain distinct IDs; inspect kNN zero distances",
            "global_number_density": len(x)/box**3, "density_units": "(cMpc/h)^-3",
            "stellar_mass_msun": _stats(mass), "log10_stellar_mass_msun": _stats(logmass),
            "stellar_particle_count": _stats(cat["stellar_particle_count"]),
            "octant_counts": np.bincount(octant, minlength=8).tolist(),
            "spatial_region_counts": np.bincount(regions, minlength=grid**3).tolist(),
            "boundary_proximity": {"width_cMpc_h": width,
                "n": int(np.any((x < width) | (x >= box-width), axis=1).sum()),
                "meaning": "diagnostic for periodic seams, NOT a physical edge or invalid record"},
            "id_note": "join on snapshot and subfind_id; IDs are not persistent across snapshots"}


def build_features(cat, dk, rho, rho2, group, labels, probs, outlier, grid, cfg):
    x, box, ids = validate_catalog(cat)
    n = len(x)
    arrays = [np.asarray(v) for v in (dk, rho, rho2, group, labels, probs, outlier)]
    if any(v.shape != (n,) for v in arrays):
        raise ValueError("derived features must preserve all catalogue rows")
    dk, rho, rho2, group, labels, probs, outlier = arrays
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < -1):
        raise ValueError("cluster labels must be categorical integers >=-1")
    if not np.isin(group, ["high", "medium", "low"]).all():
        raise ValueError("density_group must be high, medium or low")
    valid_rho = np.isfinite(rho) & (rho > 0)
    median = np.median(rho[valid_rho]) if np.any(valid_rho) else np.nan
    normalized = np.full(n, np.nan)
    percentile = np.full(n, np.nan)
    normalized[valid_rho] = rho[valid_rho]/median
    percentile[valid_rho] = rankdata(rho[valid_rho], method="average")/valid_rho.sum()
    overdensity = np.full(n, np.nan)
    overdensity[valid_rho] = rho[valid_rho]/(n/box**3)-1
    mass = np.asarray(cat["stellar_mass_msun"], dtype=float)
    logmass = np.full(n, np.nan)
    goodmass = np.isfinite(mass) & (mass > 0)
    logmass[goodmass] = np.log10(mass[goodmass])
    cells = np.minimum((x/box*int(grid)).astype(int), int(grid)-1)
    regions = cells[:, 0]*int(grid)**2 + cells[:, 1]*int(grid) + cells[:, 2]
    valid = (valid_rho & np.isfinite(dk) & (dk > 0) & np.isfinite(rho2) & (rho2 > 0)
             & goodmass & np.isfinite(probs) & (probs >= 0) & (probs <= 1)
             & np.isfinite(outlier) & (outlier >= 0) & (outlier <= 1))
    snap = int(cat["snapshot"])
    return {"coords": x, "coordinate_units": np.array("cMpc/h"),
            "subfind_id": ids, "snapshot": np.int64(snap), "redshift": np.float64(cat["redshift"]),
            "object_key": np.array([f"{snap}:{int(i)}" for i in ids]),
            "data_kind": np.array(str(cat.get("data_kind", "UNSPECIFIED"))),
            "halo_id": np.asarray(cat["halo_id"]), "stellar_mass_msun": mass,
            "log10_stellar_mass_msun": logmass,
            "stellar_particle_count": np.asarray(cat["stellar_particle_count"]),
            "knn_k": np.int64(cfg["knn_k"]), "knn_distance": dk, "rho": rho,
            "rho_over_median": normalized, "number_overdensity": overdensity,
            "density_percentile_rank": percentile, "rho_alternate_k": rho2,
            "knn_alternate_k": np.int64(cfg["knn_sensitivity_k"]),
            "density_group": group.astype("U6"), "cluster_label": labels,
            "is_noise": labels == -1, "membership_strength": probs,
            "outlier_score": outlier, "region_id": regions,
            "valid_derived_features": valid, "boxsize_cMpc_h": np.float64(box)}
