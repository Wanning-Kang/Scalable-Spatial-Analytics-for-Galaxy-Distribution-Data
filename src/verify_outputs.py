"""Independent final-output checks and checksums; no scientific verdict."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_analysis as ra
import spatial as sp

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    science = ROOT/"results/science"
    manifest = load(science/"run_manifest.json")
    cfg = manifest["configuration"]
    for name, expected in manifest["numeric_code_sha256"].items():
        assert ra.sha256(ROOT/"src"/name) == expected, f"Numeric code changed during run: {name}"
    for name, expected in manifest["source_sha256"].items():
        snap = int(name.split("_")[1])
        assert ra.sha256(ROOT/"prepared"/f"snapshot_{snap:03d}.npz") == expected
    actual = ra.version_record()
    assert actual["dependencies"] == manifest["environment"]["dependencies"]
    assert actual["python"] == manifest["environment"]["python"]
    assert load(science/"run_state.json")["status"] == "all_12_cases_computations_complete"
    checks = []
    # Rerun tests in the exact final runtime; write only the test's own log.
    command = [sys.executable, "-B", "-m", "pytest", "tests", "-q", "-rs"]
    test = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (ROOT/"results/tests_final.log").write_text(test.stdout+test.stderr)
    if test.returncode:
        raise RuntimeError("Final test suite failed")
    summary = [s for s in test.stdout.splitlines() if "passed" in s][-1]
    for snap in cfg["snapshots"]:
        q = load(ROOT/"prepared"/f"snapshot_{snap:03d}.quality.json")
        with np.load(ROOT/"prepared"/f"snapshot_{snap:03d}.npz", allow_pickle=False) as a:
            assert len(a["coords"]) == q["n_selected"]
            assert np.unique(a["subfind_id"]).size == len(a["coords"])
        for field, item in q["source_fields"].items():
            assert ra.sha256(item["path"]) == item["sha256"], f"Raw {snap}/{field} changed"
        checks.append({"check": "raw_fields_unchanged_and_prepared_identity", "snapshot": snap, "status": "passed"})
    for name, case in manifest["cases"].items():
        d = science/name
        done, s = load(d/"case_complete.json"), load(d/"summary.json")
        assert done["run_fingerprint"] == manifest["run_fingerprint"]
        assert s["n"] == case["n"] == done["n"]
        assert s["data_kind"] == case["data_kind"]
        required = ["global.npz", "points.npz", "feature_table.npz", "feature_metadata.json", "eda.json",
                    "clusters.json", "hdbscan_sensitivity.json", "random_catalog.npz",
                    "random_catalog_robustness.json", "subset_summary.json", "removal_summary.json",
                    "dd_decomposition.json", "evidence_audit.json"]
        for filename in required:
            assert (d/filename).is_file(), f"Missing {d/filename}"
        global_data, points = arrays(d/"global.npz"), arrays(d/"points.npz")
        assert global_data["jk_replicates"].shape == (27, 12)
        assert len(points["coords"]) == s["n"]
        assert np.unique(points["subfind_id"]).size == s["n"]
        assert load(d/"feature_metadata.json")["invalid_derived_rows"] == 0
        with np.load(d/"random_catalog.npz") as random:
            nr = len(random["coords"])
            assert nr == s["nr"] == int(s["n"]*cfg["random_ratio"])
        values = sp.estimators(global_data["DD"], global_data["DR"], global_data["RR"], s["n"], nr)
        for key, v in values.items():
            np.testing.assert_allclose(global_data["xi_"+key], v, equal_nan=True)
        np.testing.assert_allclose(global_data["xi"], values["landy_szalay"], equal_nan=True)
        np.testing.assert_allclose(global_data["covariance"], ra.jk_cov(global_data["jk_replicates"]), equal_nan=True)
        decomposition = load(d/"dd_decomposition.json")
        np.testing.assert_array_equal(sum(np.asarray(v) for v in decomposition["raw"].values()), global_data["DD"])
        subset_summary = load(d/"subset_summary.json")
        for label in ["density_high", "density_medium", "density_low", "cluster", "noise"]:
            sub = arrays(d/f"subset_{label}.npz")
            assert sub["DD"].shape == global_data["DD"].shape
            np.testing.assert_array_equal(sub["RR"], global_data["RR"])
            recomputed = sp.estimators(sub["DD"], sub["DR"], sub["RR"], subset_summary[label]["nd"], nr)
            np.testing.assert_allclose(sub["xi"], recomputed["landy_szalay"], equal_nan=True)
        for percent in [5, 10, 20]:
            removal = arrays(d/f"removal_{percent}.npz")
            assert removal["random_xi"].shape == (10, 12)
        # Force loading each stored member to check NPZ ZIP integrity, including
        # random coordinates and conditional curves, not just file existence.
        for p in d.glob("*.npz"):
            arrays(p)
        checks.append({"check": "case_files_arrays_counts_features_controls", "case": name, "status": "passed"})
    for snap in cfg["snapshots"]:
        b = load(ROOT/"results"/f"benchmark_{snap:03d}"/"benchmark_summary.json")
        assert b["status"] == "completed" and b["all_overlapping_counts_equal"]
        assert len(b["correctness"]) == 15 and not b["worker_failures"]
        assert all(v["same_input"] and v["exact_equal"] for v in b["correctness"])
    b = load(ROOT/"results/benchmark_engineering_100k/benchmark_summary.json")
    assert b["status"] == "completed" and b["all_overlapping_counts_equal"]
    assert any(v["method"] == "ckdtree" and v["n"] == 100000 and v["completed_repeats"] == 3 for v in b["metrics"])
    e = load(ROOT/"benchmark_catalogues/snapshot_099_scaling_100k.engineering-only.provenance.json")
    assert ra.sha256(e["output_path"]) == e["output_sha256"]
    null_dir = ROOT/"results/uniform_null_ensemble"
    for item in load(null_dir/"summary.json"):
        a = arrays(null_dir/f"snapshot_{item['snapshot']:03d}.npz")
        assert a["xi"].shape == (20, 12)
        recomputed = sp.estimators(a["DD"], a["DR"], a["RR"], item["n"], item["n"]*5)["landy_szalay"]
        np.testing.assert_allclose(recomputed, a["xi"], equal_nan=True)
    assert load(null_dir/"original_033_pair_check.json")["status"].startswith("three_small_scale_histograms_exactly_match")
    assert load(ROOT/"results/figure_review.json")["status"] == "passed"
    # Checksums cover generated scientific data, scripts, tests and presentation
    # files, excluding this self-referential report and mutable runtime caches.
    excluded = {"results/integrity_manifest.json", "results/verification.json"}
    files = []
    for p in sorted(ROOT.rglob("*")):
        relative = p.relative_to(ROOT)
        if (not p.is_file() or relative.as_posix() in excluded
                or any(x.startswith(".") or x == "__pycache__" for x in relative.parts)):
            continue
        files.append({"path": relative.as_posix(), "bytes": p.stat().st_size, "sha256": ra.sha256(p)})
    ra.save_json(ROOT/"results/integrity_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "generated analysis folder except mutable caches and self-referential verification files",
        "files": files})
    ra.save_json(ROOT/"results/verification.json", {"status": "passed", "checks": checks,
        "pytest_command": command, "pytest_summary": summary,
        "known_limits": ["Download manifests are local, not upstream independent checksums",
            "Field-only HDF5 snapshots lack headers; metadata/provenance identifies source",
            "Sparse HDBSCAN and fixed-label internal JK limitations remain",
            "One original uniform null deviates from zero; added 20-realisation checks do not fully calibrate the null tail or JK errors",
            "Numeric completion is not a causal physical finding"],
        "generated_file_count": len(files), "verified_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Verified 12 cases, 20 unchanged raw fields, benchmarks and {len(files)} generated files; {summary}")


if __name__ == "__main__":
    main()
