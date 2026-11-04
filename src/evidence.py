"""Audit saved outputs without assuming the CV's physical finding is true.

Small-scale bins are specified before running in config.json. Bin averages are
descriptive equal-bin means, not an integral of the correlation function.
Any synthetic/control run remains pending_real_data, regardless of its numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def small_scale_mask(edges, limits):
    edges = np.asarray(edges, dtype=float)
    lo, hi = map(float, limits)
    if not np.isfinite([lo, hi]).all() or lo <= 0 or hi <= lo:
        raise ValueError("small-scale interval must be finite, positive and increasing")
    return (edges[:-1] >= lo) & (edges[1:] <= hi)


def paired_contrast(first, second, first_reps, second_reps, mask):
    """Equal-bin mean contrast with matched delete-region covariance."""
    first, second = np.asarray(first), np.asarray(second)
    a, b = np.asarray(first_reps), np.asarray(second_reps)
    if first.shape != second.shape or a.shape != b.shape or a.ndim != 2 or a.shape[1:] != first.shape:
        raise ValueError("paired contrasts require matching bins and spatial replicates")
    if len(a) < 2:
        raise ValueError("at least two spatial jackknife replicates required")
    valid = (np.asarray(mask, dtype=bool) & np.isfinite(first) & np.isfinite(second)
             & np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0))
    if not valid.any():
        return {"status": "insufficient_valid_bins", "bin_indices": [], "mean_difference": None, "internal_jk_se": None}
    delta = (a-b)[:, valid].mean(axis=1)
    se = np.sqrt((len(a)-1)/len(a)*np.sum((delta-delta.mean())**2))
    return {"status": "descriptive_fixed_label_internal_uncertainty",
            "bin_indices": np.flatnonzero(valid).tolist(),
            "mean_difference": float(np.mean((first-second)[valid])),
            "internal_jk_se": float(se), "jk_regions": len(a),
            "warning": "labels fixed from full sample; not causal evidence or full pipeline resampling"}


def _curve_discrepancy(curve, reference):
    curve, reference = np.asarray(curve, dtype=float), np.asarray(reference, dtype=float)
    valid = np.isfinite(curve) & np.isfinite(reference)
    delta = curve[valid]-reference[valid]
    return {"common_finite_bins": int(valid.sum()),
            "rms_difference": float(np.sqrt(np.mean(delta**2))) if valid.any() else None,
            "max_abs_difference": float(np.max(np.abs(delta))) if valid.any() else None}


def _rank_relation(rows, field):
    x = np.array([r["median_rho"] for r in rows], dtype=float)
    y = np.array([r[field] if r[field] is not None else np.nan for r in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    usable = valid.sum() >= 3 and np.unique(x[valid]).size > 1 and np.unique(y[valid]).size > 1
    return {"n_clusters": int(valid.sum()),
            "spearman": float(spearmanr(x[valid], y[valid]).statistic) if usable else None,
            "warning": "descriptive cluster-level association; few groups can make rank estimates unstable; no p-value or causal claim"}


def audit_saved_analysis(directory, cfg):
    directory = Path(directory)
    summary = _read(directory/"summary.json")
    full = _arrays(directory/"global.npz")
    mask = small_scale_mask(full["edges"], cfg["small_scale_range_cMpc_h"])
    # Presence of TNG in a filename alone never qualifies as real data.
    real = summary["data_kind"].startswith("TNG") and int(summary["snapshot"]) in cfg["snapshots"]
    status = "descriptive_evidence_only" if real else "pending_real_data"
    ls = full["xi"]
    estimators = {k[3:]:_curve_discrepancy(v, ls) for k,v in full.items() if k.startswith("xi_")}
    mc = [{"seed": r["seed"], "random_ratio": r["random_ratio"],
           "comparison_to_primary_ls": _curve_discrepancy(r["estimators"]["landy_szalay"], ls)}
          for r in _read(directory/"random_catalog_robustness.json")]
    cov = full["covariance"]
    # NaN rows for empty bins can invalidate all rows; restrict to finite diagonal first.
    good_cov = np.isfinite(np.diag(cov))
    sub_cov = cov[np.ix_(good_cov, good_cov)]
    covariance_rank = int(np.linalg.matrix_rank(sub_cov)) if sub_cov.size and np.isfinite(sub_cov).all() else 0
    comparisons = {}
    for first, second, title in [("density_high", "density_low", "high_minus_low"),
                                 ("cluster", "noise", "cluster_minus_noise")]:
        a, b = _arrays(directory/f"subset_{first}.npz"), _arrays(directory/f"subset_{second}.npz")
        if "jk_replicates" in a and "jk_replicates" in b:
            comparisons[title] = paired_contrast(a["xi"], b["xi"], a["jk_replicates"], b["jk_replicates"], mask)
        else:
            comparisons[title] = {"status": "insufficient_objects", "mean_difference": None}
    removals = {}
    for percent, item in _read(directory/"removal_summary.json").items():
        a = _arrays(directory/f"removal_{percent}.npz")
        valid = mask & np.isfinite(a["targeted_xi"]) & np.isfinite(a["original_xi"])
        valid &= np.isfinite(a["random_xi"]).all(axis=0)
        if valid.any():
            target = float(np.mean((a["targeted_xi"]-a["original_xi"])[valid]))
            random = (a["random_xi"]-a["original_xi"][None,:])[:, valid].mean(axis=1)
            random_mean = float(np.mean(random))
            random_sd = float(np.std(random, ddof=1)) if len(random) > 1 else None
            d_cov = a["delta_covariance"][np.ix_(valid, valid)]
            weight = np.full(valid.sum(), 1/valid.sum())
            variance = weight@d_cov@weight
            se = float(np.sqrt(max(0., variance))) if np.isfinite(variance) else None
        else:
            target = random_mean = random_sd = se = None
        removals[percent] = {**item, "bin_indices": np.flatnonzero(valid).tolist(),
                             "targeted_mean_delta_xi": target, "paired_internal_jk_se": se,
                             "random_mean_delta_xi": random_mean, "random_deletion_sd": random_sd,
                             "target_minus_random_mean": target-random_mean if target is not None else None,
                             "warning": "random deletion variation is not independent cosmic uncertainty"}
    decomposition = _read(directory/"dd_decomposition.json")
    raw_all = np.asarray(decomposition["all"], dtype=np.int64)
    raw = {k:np.asarray(v, dtype=np.int64) for k,v in decomposition["raw"].items()}
    if not np.array_equal(sum(raw.values()), raw_all) or not np.array_equal(raw_all, full["DD"]):
        raise ValueError("DD decomposition does not exactly sum to the saved full raw DD")
    total = int(raw_all[mask].sum())
    dd = {}
    for kind, values in raw.items():
        observed = int(values[mask].sum())
        expected = np.asarray(decomposition["expected_random_labels"][kind], dtype=float)[mask].sum()
        dd[kind] = {"raw_pairs": observed, "raw_pair_share": observed/total if total else None,
                    "random_label_enrichment": float(observed/expected) if expected > 0 else None}
    clusters = _read(directory/"clusters.json")
    fit = summary["fit"]
    return finite_json({"data_kind": summary["data_kind"], "snapshot": summary["snapshot"],
        "physical_claim_status": status, "small_scale_range_cMpc_h": cfg["small_scale_range_cMpc_h"],
        "small_scale_bin_indices": np.flatnonzero(mask),
        "aggregation": "equal weight across bins fully inside the predeclared small-scale interval",
        "estimator_discrepancies": estimators, "random_catalog_sensitivity": mc,
        "random_catalog_warning": "finite random-catalog Monte Carlo sensitivity, not cosmic variance",
        "fit_diagnostics": {k:fit.get(k) for k in ("success", "message", "n_bins", "r0", "gamma", "rmse",
                              "reduced_chi_square", "bound_hit", "covariance_condition_number")},
        "parameter_jackknife": summary["parameter_jackknife"],
        "jackknife_diagnostics": {"regions": int(len(full["jk_replicates"])),
            "finite_covariance_bins": int(good_cov.sum()), "covariance_rank": covariance_rank,
            "singular": covariance_rank < int(good_cov.sum()),
            "warning": "internal spatial uncertainty; fixed density/cluster labels in subset/removal resamples; not box-external cosmic variance"},
        "density_rank_spearman_alternate_k": summary["density_rank_spearman"],
        "hdbscan_sensitivity": _read(directory/"hdbscan_sensitivity.json"),
        "joint_1_density_strata": comparisons["high_minus_low"],
        "joint_2_cluster_noise": comparisons["cluster_minus_noise"],
        "joint_3_density_removal": removals, "joint_4_raw_DD_sources": dd,
        "raw_DD_warning": "raw DD shares/enrichment are not fractions of Landy-Szalay excess correlation; labels were derived from the same coordinates",
        "cluster_relations_to_median_density": {key:_rank_relation(clusters, key) for key in
            ("members", "r90_cMpc_h", "compactness_descriptor", "mean_membership_strength", "cluster_persistence")},
        "claim_rule": "Synthetic data cannot support the real-galaxy finding. For real data, compare matched-N Poisson controls, mass-matched snapshots, graph/parameter sensitivity and uncertainties before drafting a descriptive conclusion; no automatic causal verdict."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"config.json")
    args = parser.parse_args()
    cfg = _read(args.config)
    for summary in sorted(args.analysis_dir.glob("*/summary.json")):
        report = audit_saved_analysis(summary.parent, cfg)
        (summary.parent/"evidence_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(f"{summary.parent.name}: {report['physical_claim_status']}", flush=True)


if __name__ == "__main__":
    main()
