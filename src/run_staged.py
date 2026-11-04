"""Resumable case-by-case real-TNG experiment, with explicit control identities.

Run after prepare_tng.py. Original raw HDF5 inputs are never written. A case
is resumable only if its completion marker matches numeric code and inputs.
Partial case directories stop the run; no old output is silently replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

import run_analysis as ra
import spatial as sp

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def catalogues(prepared, cfg):
    cats = []
    for snap in cfg["snapshots"]:
        path = prepared / f"snapshot_{snap:03d}.npz"
        with np.load(path, allow_pickle=False) as data:
            cat = {k: data[k] for k in data.files}
        if int(cat["snapshot"]) != snap:
            raise ValueError("Prepared snapshot identity mismatch")
        cat.update(data_kind="TNG100-1_SELECTED_GALAXY_HOSTING_SUBHALOS",
                   source_path=path.name, source_sha256=ra.sha256(path))
        cats.append(cat)
    matched, matching = ra.mass_matched_catalogues(cats, cfg)
    cases = {}
    for cat, sub in zip(cats, matched):
        name = f"snapshot_{int(cat['snapshot']):03d}"
        cases[name] = cat
        if cfg["control_poisson"]:
            control = dict(cat)
            control["coords"] = np.random.default_rng(cfg["seed"]+int(cat["snapshot"])+10000).uniform(
                0, float(cat["boxsize_cMpc_h"]), (len(cat["coords"]), 3))
            control["halo_id"] = np.arange(len(cat["coords"]))
            control["data_kind"] = "SYNTHETIC_UNIFORM_MATCHED_N_CONTROL"
            cases[name+"_poisson_control"] = control
        cases[name+"_mass_matched"] = sub
    return cases, matching


def fingerprint(cfg_path, cases):
    files = [ROOT/"src"/name for name in ["spatial.py", "run_analysis.py", "feature_audit.py", "evidence.py", "run_staged.py"]]
    info = {"config_sha256": ra.sha256(cfg_path),
            "numeric_code_sha256": {p.name: ra.sha256(p) for p in files},
            "source_sha256": {name: cat["source_sha256"] for name, cat in cases.items()}}
    return hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest(), info


def graph_preflight(cat, cfg):
    x = np.asarray(cat["coords"], dtype=float)
    if len(np.unique(x, axis=0)) != len(x):
        raise ValueError("Duplicate selected coordinates need a declared clustering policy")
    tree = cKDTree(x, boxsize=float(cat["boxsize_cMpc_h"]))
    records = []
    q = max(cfg["graph_k"], cfg["min_samples"])
    maximum = min(cfg["max_graph_k"], len(x)-1)
    while q <= maximum:
        estimate = len(x)*(q+1)*160
        if estimate > 1_500_000_000:
            raise MemoryError("Periodic graph estimate exceeds the existing 1.5 GB guard")
        graph = sp._sparse_periodic_graph(tree, x, q)
        n_components = connected_components(graph, directed=False, return_labels=False)
        degree = np.diff(graph.indptr)
        records.append({"k": q, "components": int(n_components), "minimum_degree": int(degree.min()),
                        "directed_edges": int(graph.nnz), "estimated_buffer_bytes": estimate})
        if n_components == 1 and degree.min() >= cfg["min_samples"]:
            return {"n": len(x), "status": "passed", "attempts": records}
        if q == maximum:
            break
        q = min(maximum, q*2)
    raise RuntimeError(f"Periodic graph preflight failed: {records}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT/"prepared")
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results"/"science")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cases, matching = catalogues(args.prepared_dir, cfg)
    run_hash, info = fingerprint(args.config, cases)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.case:
        cat = cases[args.case]
        target = out / args.case
        if target.exists():
            raise FileExistsError(f"Case output already exists: {target}")
        ra.save_json(out/"diagnostics"/(args.case+"_graph_preflight.json"), graph_preflight(cat, cfg))
        start = time.perf_counter()
        ra.analyze(cat, cfg, target)
        ra.save_json(target/"case_complete.json", {"status": "computations_complete_not_a_causal_verdict",
                     "run_fingerprint": run_hash, "data_kind": cat["data_kind"], "n": len(cat["coords"]),
                     "source_sha256": cat["source_sha256"], "completed_utc": stamp(),
                     "elapsed_seconds": time.perf_counter()-start,
                     "whole_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "peak_rss_platform_units": "bytes on macOS"})
        return 0
    manifest_path = out/"run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["run_fingerprint"] != run_hash:
            raise ValueError("Code/config/input changed; use a NEW output directory")
    else:
        ra.save_json(manifest_path, {"mode": "REAL_TNG_WITH_EXPLICIT_SYNTHETIC_CONTROLS", "started_utc": stamp(),
            "run_fingerprint": run_hash, **info, "configuration": cfg, "environment": ra.version_record(),
            "cases": {k: {"n": len(v["coords"]), "data_kind": v["data_kind"], "source_path": v["source_path"]}
                      for k, v in cases.items()},
            "notes": ["No API key required or recorded", "Raw inputs read-only", "Uniform controls are not real galaxies",
                      "Numerical completion does not confirm the CV hypothesis"]})
    ra.save_json(out/"mass_matching.json", matching)
    selected = args.only or list(cases)
    if any(name not in cases for name in selected):
        parser.error("Unknown case")
    for name in selected:
        complete = out/name/"case_complete.json"
        if complete.exists():
            if json.loads(complete.read_text())["run_fingerprint"] != run_hash:
                raise ValueError(f"Completion fingerprint mismatch: {name}")
            print(f"Verified completed case: {name}", flush=True)
            continue
        if (out/name).exists():
            raise FileExistsError(f"Partial output preserved at {out/name}; inspect it before retrying into a new directory")
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--prepared-dir", str(args.prepared_dir),
                   "--config", str(args.config), "--output-dir", str(out), "--case", name]
        record = {"case": name, "status": "running", "started_utc": stamp(),}
        ra.save_json(out/"run_state.json", record)
        print(f"Starting case {name}: N={len(cases[name]['coords'])}", flush=True)
        (out/"logs").mkdir(exist_ok=True)
        with (out/"logs"/(name+".log")).open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        record.update(status="completed" if result.returncode == 0 else "failed", return_code=result.returncode,
                      finished_utc=stamp())
        ra.save_json(out/"run_state.json", record)
        if result.returncode:
            print(f"Stopped {name}; inspect {out/'logs'/(name+'.log')}", flush=True)
            return result.returncode
        print(f"Completed case {name}", flush=True)
    names = [f"snapshot_{s:03d}" for s in cfg["snapshots"]]
    if all((out/n/"case_complete.json").exists() for n in cases):
        summaries = [json.loads((out/n/"summary.json").read_text()) for n in names]
        ra.save_json(out/"overview.json", summaries)
        differences = []
        for suffix in ["", "_mass_matched"]:
            reference = names[0]+suffix
            with np.load(out/reference/"global.npz") as a:
                for name in names[1:]:
                    with np.load(out/(name+suffix)/"global.npz") as b:
                        differences.append({"reference": reference, "comparison": name+suffix,
                            "delta_xi": b["xi"]-a["xi"],
                            "paired_jk_covariance": ra.jk_cov(b["jk_replicates"]-a["jk_replicates"]),
                            "warning": "paired internal spatial uncertainty, not independent cosmic variance"})
        ra.save_json(out/"cross_snapshot_differences.json", differences)
        ra.save_json(out/"run_state.json", {"status": "all_12_cases_computations_complete", "finished_utc": stamp(),
                                           "run_fingerprint": run_hash})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
