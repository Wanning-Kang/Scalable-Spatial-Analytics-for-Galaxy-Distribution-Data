"""Predeclared 20-seed global-only uniform null ensemble for each real N.

This supplements, NEVER replaces, the original matched-N control cases.
Both D and R are independently regenerated for each deterministic seed. No
HDBSCAN or repeated model fitting; no seed selection based on the outcome.
"""
from pathlib import Path
import json
import time

import numpy as np

import run_analysis as ra
import spatial as sp
from evidence import small_scale_mask

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT/"config.json").read_text())
    target = ROOT/"results/uniform_null_ensemble"
    if target.exists():
        raise FileExistsError("Preserve the existing ensemble; use a new directory for another experiment")
    target.mkdir(parents=True)
    base_seed, repeats = 20261021, 20
    edges = np.asarray(cfg["edges_cMpc_h"])
    mask = small_scale_mask(edges, cfg["small_scale_range_cMpc_h"])
    ra.save_json(target/"design.json", {"repeats": repeats, "base_seed": base_seed,
        "seed_rule": "SeedSequence([20261021,snapshot,repeat,2026]) with two spawned streams for D and R",
        "n_random": "5*N", "scope": "global-only finite-sample and finite-random-catalogue null variation",
        "original_control_preserved": True, "selection": "all twenty outcomes retained; no early stopping or selected favorable seed",
        "configuration_sha256": ra.sha256(ROOT/"config.json"), "spatial_sha256": ra.sha256(ROOT/"src/spatial.py")})
    summaries = []
    for snap in cfg["snapshots"]:
        with np.load(ROOT/"prepared"/f"snapshot_{snap:03d}.npz") as f:
            n, box = len(f["coords"]), float(f["boxsize_cMpc_h"])
        xi, dd, dr, rr, seeds = [], [], [], [], []
        st = time.perf_counter()
        for repeat in range(repeats):
            seed = np.random.SeedSequence([base_seed, snap, repeat, 2026])
            first, second = seed.spawn(2)
            data = np.random.default_rng(first).uniform(0, box, (n, 3))
            random = np.random.default_rng(second).uniform(0, box, (5*n, 3))
            d = sp.pair_counts(data, None, edges, box)
            c = sp.pair_counts(data, random, edges, box)
            r = sp.pair_counts(random, None, edges, box)
            dd.append(d); dr.append(c); rr.append(r)
            xi.append(sp.estimators(d, c, r, n, 5*n)["landy_szalay"])
            seeds.append([base_seed, snap, repeat, 2026])
        xi = np.asarray(xi)
        values = np.mean(xi[:, mask], axis=1)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        original_path = ROOT/"results/science"/f"snapshot_{snap:03d}_poisson_control/global.npz"
        with np.load(original_path) as f:
            original = float(np.mean(f["xi"][mask]))
            w = np.full(mask.sum(), 1/mask.sum())
            original_se = float(np.sqrt(w @ f["covariance"][np.ix_(mask, mask)] @ w))
        np.savez_compressed(target/f"snapshot_{snap:03d}.npz", edges=edges, xi=xi, DD=dd, DR=dr, RR=rr,
                            seed_components=seeds, small_scale_means=values)
        result = {"snapshot": snap, "n": n, "repeats": repeats,
            "mean_small_scale_xi": mean, "between_realisation_sd": sd,
            "mean_monte_carlo_se": sd/np.sqrt(repeats),
            "approx_95_normal_interval_for_ensemble_mean": [mean-1.96*sd/np.sqrt(repeats), mean+1.96*sd/np.sqrt(repeats)],
            "original_single_control_mean": original, "original_single_control_internal_jk_se": original_se,
            "original_distance_from_ensemble_mean_in_sd": (original-mean)/sd,
            "minimum_over_twenty": float(values.min()), "maximum_over_twenty": float(values.max()),
            "negative_mean_count": int((values < 0).sum()),
            "elapsed_seconds": time.perf_counter()-st,
            "warning": "20 repetitions describe this null Monte Carlo distribution, not independent cosmic variance; do not turn this into a calibrated tail probability or replace the original control"}
        summaries.append(result)
        print(f"snapshot {snap}: null ensemble {mean:.4f} +/- {sd:.4f} between-realisation SD; original {original:.4f}", flush=True)
    ra.save_json(target/"summary.json", summaries)


if __name__ == "__main__":
    main()
