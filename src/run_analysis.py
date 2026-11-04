"""Run the documented experiment, keeping synthetic validation separate from TNG.

The full original cleaned catalogue supplies both global and local analysis.
Derived density/cluster subsets do NOT define new physical observations.
No raw catalogue is modified. Re-runs require a new empty output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

import spatial as sp
from feature_audit import audit_catalog, build_features
from evidence import audit_saved_analysis


ROOT = Path(__file__).resolve().parents[1]


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def version_record():
    names = ["numpy", "scipy", "hdbscan", "scikit-learn", "h5py", "matplotlib"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {name: importlib.metadata.version(name) for name in names}}


def synthetic_catalog(seed, n=1200, box=75., clustered=True):
    """Deliberately artificial clustered + background case, NEVER TNG data."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, box, (n, 3))
    truth = np.full(n, -1, dtype=int)
    if clustered:
        centers = np.array([[0.1, 0.2, 0.1], [26, 29, 33], [49, 51, 48]])
        count = min(240, max(2, n//5))
        for j, center in enumerate(centers):
            start = j * count
            x[start:start+count] = (center + rng.normal(0, [.35, .65, .9][j], (count, 3))) % box
            truth[start:start+count] = j
    return {"coords": x, "subfind_id": np.arange(n), "halo_id": truth,
            "stellar_mass_msun": 10**rng.uniform(9, 11.2, n),
            "stellar_particle_count": np.full(n, 300), "snapshot": -1,
            "redshift": np.nan, "boxsize_cMpc_h": box, "coordinate_units": "cMpc/h",
            "data_kind": "SYNTHETIC_CLUSTERED" if clustered else "SYNTHETIC_UNIFORM"}


def ls_sample(x, random, edges, box, rr=None):
    if len(x) < 2:
        return {"DD": np.zeros(len(edges)-1, dtype=np.int64),
                "DR": np.zeros(len(edges)-1, dtype=np.int64),
                "RR": rr if rr is not None else sp.pair_counts(random, None, edges, box),
                "xi": np.full(len(edges)-1, np.nan), "nd": len(x), "nr": len(random)}
    dd = sp.pair_counts(x, None, edges, box)
    dr = sp.pair_counts(x, random, edges, box)
    rr = sp.pair_counts(random, None, edges, box) if rr is None else rr
    est = sp.estimators(dd, dr, rr, len(x), len(random))
    return {"DD": dd, "DR": dr, "RR": rr, "xi": est["landy_szalay"],
            "estimators": est, "nd": len(x), "nr": len(random)}


def jk_cov(reps):
    reps = np.asarray(reps)
    valid = np.all(np.isfinite(reps), axis=0)
    cov = np.full((reps.shape[1], reps.shape[1]), np.nan)
    delta = reps[:, valid] - np.mean(reps[:, valid], axis=0)
    cov[np.ix_(valid, valid)] = (len(reps)-1)/len(reps) * delta.T @ delta
    return cov


def parameter_jackknife_summary(fit, jk_fits):
    """Report internal parameter uncertainty only for comparable interior fits.

    A delete-region fit must use exactly the full-sample fitting bins. A
    numerically successful bound-constrained fit is not an interior estimate
    suitable for the normal-interval approximation used here.
    """
    reason = None
    if not fit.get("success") or not all(v.get("success") for v in jk_fits):
        reason = "one_or_more_fits_failed"
    elif len(jk_fits) < 2:
        reason = "fewer_than_two_delete_region_fits"
    elif fit.get("bound_hit", True) or any(v.get("bound_hit", True) for v in jk_fits):
        reason = "one_or_more_fits_hit_parameter_bounds"
    else:
        indices = np.asarray(fit.get("fit_indices", []), dtype=int)
        if not len(indices) or any(not np.array_equal(
                indices, np.asarray(v.get("fit_indices", []), dtype=int)) for v in jk_fits):
            reason = "delete_region_fitting_bins_differ_from_full_sample"
    if reason is not None:
        return {"status": "unavailable", "reason": reason,
                "warning": "No parameter interval reported: all fits must succeed, use identical bins, and remain off bounds."}
    center = np.asarray([fit["r0"], fit["gamma"]], dtype=float)
    values = np.asarray([[v["r0"], v["gamma"]] for v in jk_fits], dtype=float)
    if not np.isfinite(center).all() or not np.isfinite(values).all():
        return {"status": "unavailable", "reason": "nonfinite_fitted_parameters"}
    delta = values-values.mean(axis=0)
    covariance = (len(values)-1)/len(values)*delta.T@delta
    se = np.sqrt(np.diag(covariance))
    return {"status": "approximate_internal_spatial_uncertainty", "covariance": covariance,
            "r0_se": se[0], "gamma_se": se[1], "fit_indices": indices,
            "jk_regions": len(values),
            "approx_95_normal_intervals": np.column_stack((center-1.96*se, center+1.96*se)),
            "warning": "Approximate normal internal-spatial intervals, not percentile bootstrap or independent cosmic-variance intervals; negative bounds are not physically meaningful."}


def graph_k_sensitivity(x, box, cfg, primary):
    """Compare distinct *actual* sparse graph truncations, or report why not.

    The primary graph may already have escalated above cfg['graph_k'] to
    become connected. Starting from twice that configured value can silently
    reproduce the same graph; the comparison must exceed the actual value.
    """
    metadata = primary["metadata"]
    result = {"status": "unavailable", "primary": metadata, "higher": None,
              "requested_graph_k": None,
              "warning": "Graph-k sensitivity is not proof of exact complete-metric HDBSCAN."}
    actual = metadata.get("graph_k")
    if actual is None or not metadata.get("approximation", False):
        result["reason"] = "primary_uses_exact_dense_distances"
        return result
    actual = int(actual)
    maximum = min(int(cfg["max_graph_k"]), len(x)-1)
    requested = min(maximum, max(actual+1, 2*actual))
    if requested <= actual:
        result["reason"] = "no_larger_graph_k_within_configured_limit"
        return result
    result["requested_graph_k"] = requested
    try:
        higher = sp.periodic_hdbscan(x, box, cfg["min_cluster_size"], cfg["min_samples"],
                                    graph_k=requested, max_graph_k=maximum, dense_limit=0)
    except (MemoryError, RuntimeError) as exc:
        result.update(reason="larger_graph_unavailable", error_type=type(exc).__name__,
                      error=str(exc))
        return result
    result["higher"] = higher["metadata"]
    higher_actual = higher["metadata"].get("graph_k")
    if higher_actual is None or int(higher_actual) <= actual:
        result["reason"] = "returned_graph_k_is_not_larger_than_primary"
        return result
    result.update(status="available",
                  ari_including_noise=float(adjusted_rand_score(primary["labels"], higher["labels"])),
                  noise_agreement=float(np.mean((primary["labels"] == -1) == (higher["labels"] == -1))))
    return result


def density_groups(rho, ids):
    """Exact 20/60/20 ranks; Subfind ID breaks ties reproducibly."""
    order = np.lexsort((ids, rho))
    groups = np.full(len(rho), "medium", dtype="U6")
    tail = int(np.floor(.2 * len(rho)))
    groups[order[:tail]] = "low"
    if tail:
        groups[order[-tail:]] = "high"
    return groups


def cluster_table(x, rho, labels, probabilities, persistence, halo, box):
    rows = []
    for label in np.unique(labels[labels >= 0]):
        keep = labels == label
        angle = x[keep]/box * 2*np.pi
        center = (np.arctan2(np.mean(np.sin(angle), axis=0),
                             np.mean(np.cos(angle), axis=0)) % (2*np.pi)) * box/(2*np.pi)
        delta = np.abs(x[keep]-center)
        radius = np.linalg.norm(np.minimum(delta, box-delta), axis=1)
        r90 = float(np.quantile(radius, .9))
        _, counts = np.unique(halo[keep], return_counts=True)
        rows.append({"cluster_label": int(label), "members": int(np.sum(keep)),
                     "median_rho": float(np.median(rho[keep])), "center_cMpc_h": center,
                     "r90_cMpc_h": r90, "compactness_descriptor": np.sum(keep)/(4*np.pi/3*r90**3) if r90 else np.nan,
                     "mean_membership_strength": float(np.mean(probabilities[keep])),
                     "cluster_persistence": float(persistence[label]) if label < len(persistence) else None,
                     "dominant_fof_fraction": float(np.max(counts)/np.sum(keep))})
    return rows


def subset_analysis(x, masks, random, edges, box, cfg, full, out):
    grid = cfg["jackknife_grid"]
    records = {}
    for name, keep in masks.items():
        sample = ls_sample(x[keep], random, edges, box, full["RR"])
        records[name] = {"nd": len(x[keep])}
        if len(x[keep]) >= 2:
            jk = sp.jackknife(x[keep], random, edges, box, grid,
                              random_cache=full["jk_cache"])
            cov = jk["covariance"]
            fit = sp.fit_powerlaw(edges, sample["xi"], cov, cfg["fit_range_cMpc_h"])
            delta_rep = jk["replicates"] - full["jk_replicates"]
            delta_cov = jk_cov(delta_rep)
            np.savez_compressed(out / f"subset_{name}.npz", edges=edges,
                                xi=sample["xi"], DD=sample["DD"], DR=sample["DR"], RR=sample["RR"],
                                jk_replicates=jk["replicates"], covariance=cov,
                                delta_vs_full=sample["xi"]-full["xi"], delta_covariance=delta_cov)
            records[name].update({"fit": fit, "valid_xi_bins": int(np.isfinite(sample["xi"]).sum())})
        else:
            records[name]["status"] = "insufficient_objects"
            np.savez_compressed(out / f"subset_{name}.npz", edges=edges, xi=sample["xi"], DD=sample["DD"], DR=sample["DR"], RR=sample["RR"])
    save_json(out / "subset_summary.json", records)
    return records


def analyze(cat, cfg, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    x = np.asarray(cat["coords"], dtype=float)
    n = len(x)
    box = float(cat["boxsize_cMpc_h"])
    if n <= max(cfg["knn_k"], cfg["min_cluster_size"]):
        raise ValueError("too few selected galaxies for configured local analysis")
    if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x >= box):
        raise ValueError("coords must lie within the finite periodic box")
    save_json(out/"eda.json", audit_catalog(cat, cfg))
    edges = np.array(cfg["edges_cMpc_h"])
    if edges[-1] >= box/2:
        raise ValueError("shell-average fitting assumes edges < L/2")
    rng = np.random.default_rng(cfg["random_seeds"][0])
    random = rng.uniform(0, box, (int(cfg["random_ratio"]*n), 3))
    np.savez_compressed(out / "random_catalog.npz", coords=random, region_id=sp.region_ids(random, box, cfg["jackknife_grid"]), seed=cfg["random_seeds"][0])
    print(f"[{out.name}] N={n}; DD/DR/RR and jackknife", flush=True)
    full = ls_sample(x, random, edges, box)
    jk = sp.jackknife(x, random, edges, box, cfg["jackknife_grid"])
    full["jk_replicates"] = jk["replicates"]
    full["jk_cache"] = jk
    fit = sp.fit_powerlaw(edges, full["xi"], jk["covariance"], cfg["fit_range_cMpc_h"])
    # Refit delete-one catalogues using the same covariance/fit rule, not percentile CIs.
    jk_fits = [sp.fit_powerlaw(edges, v, jk["covariance"], cfg["fit_range_cMpc_h"]) for v in jk["replicates"]]
    parameter_jk = parameter_jackknife_summary(fit, jk_fits)
    np.savez_compressed(out / "global.npz", edges=edges, DD=full["DD"], DR=full["DR"], RR=full["RR"], xi=full["xi"],
                        covariance=jk["covariance"], se=jk["se"], jk_replicates=jk["replicates"], **{f"xi_{key}":val for key,val in full["estimators"].items()})
    robustness = []
    for ratio, seed in [(cfg["random_ratio"], s) for s in cfg["random_seeds"][1:]] + [(2, cfg["seed"]+100)]:
        r2 = np.random.default_rng(seed).uniform(0, box, (int(n*ratio), 3))
        result = ls_sample(x, r2, edges, box)
        robustness.append({"seed": seed, "random_ratio": ratio, "estimators": result["estimators"]})
    save_json(out / "random_catalog_robustness.json", robustness)
    print(f"[{out.name}] local density and periodic HDBSCAN", flush=True)
    distance, rho = sp.density_knn(x, box, cfg["knn_k"])
    distance2, rho2 = sp.density_knn(x, box, cfg["knn_sensitivity_k"])
    group = density_groups(rho, cat["subfind_id"])
    cl = sp.periodic_hdbscan(x, box, cfg["min_cluster_size"], cfg["min_samples"],
                              graph_k=cfg["graph_k"], max_graph_k=cfg["max_graph_k"], dense_limit=cfg["dense_cluster_limit"])
    labels = cl["labels"]
    probabilities = cl["probabilities"]
    persistence = cl["persistence"]
    features = build_features(cat, distance, rho, rho2, group, labels, probabilities,
                              cl["outlier"], cfg["jackknife_grid"], cfg)
    np.savez_compressed(out/"feature_table.npz", **features)
    save_json(out/"feature_metadata.json", {
        "row_count": n, "invalid_derived_rows": int((~features["valid_derived_features"]).sum()),
        "keys": list(features), "join_key": ["snapshot", "subfind_id"],
        "coordinate_units": "cMpc/h", "density_units": "(cMpc/h)^-3",
        "rho_over_median": "density divided by the within-snapshot median, NOT conventional overdensity",
        "number_overdensity": "rho/(N/L^3)-1; finite-k k/V proxy retains estimator bias",
        "cluster_label": "categorical integer; -1 means algorithm noise, not a data error",
        "membership_strength": "HDBSCAN strength, not calibrated physical membership probability",
        "geometry": "original physical scale retained; XYZ not z-scored"})
    np.savez_compressed(out / "points.npz", coords=x, subfind_id=cat["subfind_id"], halo_id=cat["halo_id"],
                        stellar_mass_msun=cat["stellar_mass_msun"], knn_distance=distance, rho=rho,
                        rho_over_median=rho/np.median(rho), rho_alternate_k=rho2,
                        density_group=group, cluster_label=labels, is_noise=labels==-1,
                        membership_strength=probabilities, outlier_score=cl["outlier"],
                        region_id=sp.region_ids(x, box, cfg["jackknife_grid"]),
                        snapshot=cat["snapshot"], redshift=cat["redshift"])
    rows = cluster_table(x, rho, labels, probabilities, persistence, cat["halo_id"], box)
    save_json(out / "clusters.json", rows)
    sensitivity = []
    for size in [max(cfg["min_samples"]+1, cfg["min_cluster_size"]//2), cfg["min_cluster_size"]*2]:
        alt = sp.periodic_hdbscan(x, box, size, cfg["min_samples"], graph_k=cfg["graph_k"], max_graph_k=cfg["max_graph_k"], dense_limit=cfg["dense_cluster_limit"])
        sensitivity.append({"min_cluster_size": size, "ari_including_noise": adjusted_rand_score(labels, alt["labels"]),
                            "noise_fraction": np.mean(alt["labels"]==-1), "metadata": alt["metadata"]})
    # Validate the periodic sparse approximation on the SAME small point set.
    check_index = np.random.default_rng(cfg["seed"]+77).choice(n, min(n, 500), replace=False)
    small = x[check_index]
    exact = sp.periodic_hdbscan(small, box, max(10, cfg["min_cluster_size"]//2), cfg["min_samples"], dense_limit=1500)
    sparse = sp.periodic_hdbscan(small, box, max(10, cfg["min_cluster_size"]//2), cfg["min_samples"], graph_k=cfg["graph_k"], max_graph_k=cfg["max_graph_k"], dense_limit=0)
    graph_check = {"sample_n": len(small), "ari_including_noise": adjusted_rand_score(exact["labels"], sparse["labels"]),
                   "noise_agreement": float(np.mean((exact["labels"]==-1)==(sparse["labels"]==-1))),
                   "exact": exact["metadata"], "sparse": sparse["metadata"]}
    graph_sensitivity = graph_k_sensitivity(x, box, cfg, cl)
    graph_check["full_graph_sensitivity"] = graph_sensitivity
    # Keep the legacy scalar for readers, but never emit a fake ARI=1 when
    # the configured cap or resource guard prevents a distinct comparison.
    graph_check["full_graph_sensitivity_ari"] = graph_sensitivity.get("ari_including_noise")
    save_json(out / "hdbscan_sensitivity.json", {"parameters": sensitivity, "graph_validation": graph_check, "primary": cl["metadata"]})
    print(f"[{out.name}] four joint analyses", flush=True)
    masks = {f"density_{name}": group==name for name in ["high", "medium", "low"]}
    masks.update({"cluster":labels>=0, "noise":labels==-1})
    subset_summary = subset_analysis(x, masks, random, edges, box, cfg, full, out)
    removal = {}
    order = np.lexsort((cat["subfind_id"], -rho))
    delete_rng = np.random.default_rng(cfg["seed"]+301)
    for fraction in cfg["removal_fractions"]:
        deleted_n = int(np.floor(fraction*n))
        keep = np.ones(n, dtype=bool)
        keep[order[:deleted_n]] = False
        selected = ls_sample(x[keep], random, edges, box, full["RR"])
        selected_jk = sp.jackknife(x[keep], random, edges, box, cfg["jackknife_grid"],
                                   random_cache=jk)
        random_reps = []
        for j in range(cfg["random_removal_repeats"]):
            random_keep = np.ones(n, dtype=bool)
            random_keep[delete_rng.choice(n, deleted_n, replace=False)] = False
            random_reps.append(ls_sample(x[random_keep], random, edges, box, full["RR"])["xi"])
        key = str(round(100*fraction))
        np.savez_compressed(out / f"removal_{key}.npz", edges=edges, targeted_xi=selected["xi"],
                            random_xi=random_reps, original_xi=full["xi"],
                            delta_vs_full=selected["xi"]-full["xi"],
                            delta_covariance=jk_cov(selected_jk["replicates"]-jk["replicates"]))
        removal[key] = {"deleted_n":deleted_n, "remaining_n":n-deleted_n,
                        "random_repeats":cfg["random_removal_repeats"], "density_reference":"fixed original full sample"}
    save_json(out / "removal_summary.json", removal)
    parts = sp.decomposition(x, labels, edges, box)
    save_json(out / "dd_decomposition.json", parts)
    summary = {"data_kind":str(cat["data_kind"]), "snapshot":int(cat["snapshot"]), "redshift":float(cat["redshift"]),
               "n": n, "nr":len(random), "boxsize_cMpc_h":box,
               "n_clusters":len(rows), "noise_fraction":float(np.mean(labels==-1)),
               "rho_quantiles":dict(zip(["P10","P50","P90","P99"], np.quantile(rho, [.1,.5,.9,.99]))),
               "density_rank_spearman":float(spearmanr(rho,rho2).statistic),
               "fit":fit, "jk_parameter_fits":jk_fits, "parameter_jackknife":parameter_jk,
               "subsets":subset_summary, "hdbscan_metadata":cl["metadata"],
               "elapsed_seconds":time.perf_counter()-start,
               "interpretation_limits":["coordinate-derived selection induces clustering",
                                        "DD fractions are raw pair shares, not excess-correlation contributions",
                                        "noise is not a data-error diagnosis",
                                        "jackknife estimates internal spatial uncertainty, not independent cosmic variance"]}
    save_json(out / "summary.json", summary)
    save_json(out / "evidence_audit.json", audit_saved_analysis(out, cfg))
    return summary


def mass_matched_catalogues(catalogues, cfg):
    """Deterministic common mass-histogram selection, additional to full samples."""
    edges = np.asarray(cfg["cross_redshift_mass_bins_log10"])
    assignment = []
    for cat in catalogues:
        mass = np.log10(cat["stellar_mass_msun"])
        bins = np.searchsorted(edges, mass, side="right")-1
        bins[mass==edges[-1]] = len(edges)-2
        assignment.append(bins)
    numbers = np.array([[np.sum(v==i) for i in range(len(edges)-1)] for v in assignment])
    target = np.min(numbers, axis=0)
    output = []
    for cat, bins in zip(catalogues, assignment):
        rng = np.random.default_rng(cfg["seed"]+int(cat["snapshot"]))
        keep = np.sort(np.concatenate([rng.choice(np.where(bins==i)[0], int(amount), replace=False) for i,amount in enumerate(target)]))
        sub = {k: v[keep] if isinstance(v, np.ndarray) and v.ndim>0 and len(v)==len(bins) else v for k,v in cat.items()}
        sub["data_kind"] = "TNG_MASS_MATCHED_SUBSET"
        output.append(sub)
    return output, {"edges_log10_Msun":edges, "original_histograms":numbers, "matched_counts":target,
                    "warning":"coarse-bin mass matching does not match within-bin masses or prove sample completeness"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT/"data"/"prepared")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--demo", action="store_true", help="only artificial validation, never TNG findings")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be new or empty (protects previous results)")
    cats = []
    if args.demo:
        cats = [synthetic_catalog(cfg["seed"], clustered=True), synthetic_catalog(cfg["seed"]+1, clustered=False)]
        names = ["synthetic_clustered", "synthetic_uniform"]
    else:
        files = [args.prepared_dir/f"snapshot_{snap:03d}.npz" for snap in cfg["snapshots"]]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            parser.error("TNG prepared data missing; run download_tng.py and prepare_tng.py first. " + "; ".join(missing))
        for snap, path in zip(cfg["snapshots"], files):
            with np.load(path, allow_pickle=False) as file:
                cat = {key:file[key] for key in file.files}
            if int(cat["snapshot"]) != snap:
                raise ValueError(f"snapshot mismatch {path}")
            cat["data_kind"] = "TNG100-1_SELECTED_GALAXY_HOSTING_SUBHALOS"
            cat["source_path"] = path.name
            cat["source_sha256"] = sha256(path)
            cats.append(cat)
        names = [f"snapshot_{snap:03d}" for snap in cfg["snapshots"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir/"run_manifest.json", {"mode":"SYNTHETIC_VALIDATION" if args.demo else "TNG_EXPERIMENT",
              "configuration":cfg, "configuration_sha256":sha256(args.config), "environment":version_record(),
              "code_sha256":{p.name:sha256(p) for p in sorted((ROOT/"src").glob("*.py"))},
              "source_catalogues":[{k:cat[k] for k in ["source_path","source_sha256"] if k in cat} for cat in cats]})
    summaries = []
    for name,cat in zip(names,cats):
        print(f"Starting {name}", flush=True)
        summaries.append(analyze(cat,cfg,args.output_dir/name))
        if not args.demo and cfg["control_poisson"]:
            control = dict(cat)
            control["coords"] = np.random.default_rng(cfg["seed"]+int(cat["snapshot"])+10000).uniform(0,float(cat["boxsize_cMpc_h"]), (len(cat["coords"]),3))
            control["halo_id"] = np.arange(len(cat["coords"]))
            control["data_kind"] = "SYNTHETIC_UNIFORM_MATCHED_N_CONTROL"
            analyze(control,cfg,args.output_dir/(name+"_poisson_control"))
    if not args.demo:
        matched, record = mass_matched_catalogues(cats,cfg)
        save_json(args.output_dir/"mass_matching.json",record)
        for name, cat in zip(names,matched):
            analyze(cat,cfg,args.output_dir/(name+"_mass_matched"))
        # Shared periodic regions + shared first-seed random points make paired comparisons reproducible.
        comparisons = []
        for name in names[1:]:
            with np.load(args.output_dir/names[0]/"global.npz") as a, np.load(args.output_dir/name/"global.npz") as b:
                diff = b["jk_replicates"]-a["jk_replicates"]
                comparisons.append({"reference":names[0], "comparison":name,
                                    "delta_xi":b["xi"]-a["xi"], "paired_jk_covariance":jk_cov(diff)})
        save_json(args.output_dir/"cross_snapshot_differences.json",comparisons)
    save_json(args.output_dir/"overview.json",summaries)
    print(f"Saved {args.output_dir}",flush=True)


if __name__ == "__main__":
    main()
