#!/usr/bin/env python3
"""Isolated-process, reproducible spatial pair-count benchmarks.

The default input is uniform synthetic data, not a TNG galaxy catalogue. Each
method receives identical arrays for a given (N, repeat). Timing excludes data
generation/loading and imports, but peak RSS covers the whole worker process.
The tree total includes building both D and R trees and all DD/DR/RR queries.
The bounded-memory brute-force baseline does not allocate an N-by-N matrix.
Bins follow the experiment convention: (left edge, right edge].
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

# Do not use PYTHONPATH: the requested Desktop directory contains a colon.
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_SIZES = [500, 1000, 5000, 10000, 50000, 100000]
CATALOG_KEYS = ("coords", "boxsize_cMpc_h", "snapshot", "subfind_id", "coordinate_units")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def peak_rss_bytes() -> int:
    """ru_maxrss is bytes on macOS and KiB on Linux/BSD Python platforms."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_catalog(path: str | Path, default_box: float | None = None) -> tuple[np.ndarray, float]:
    """Read explicit prepared units/IDs only; default_box never repairs metadata."""
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in CATALOG_KEYS if key not in data]
        if missing:
            raise ValueError(f"Prepared catalogue metadata missing: {missing}; run prepare_tng.py")
        positions = np.asarray(data["coords"], dtype=np.float64)
        box = float(data["boxsize_cMpc_h"])
        ids = np.asarray(data["subfind_id"])
        if str(data["coordinate_units"]) != "cMpc/h":
            raise ValueError("Expected explicit coordinate_units=cMpc/h, not raw TNG ckpc/h")
        snapshot = np.asarray(data["snapshot"])
        if snapshot.ndim != 0 or not np.issubdtype(snapshot.dtype, np.integer):
            raise ValueError("snapshot must be an integer scalar")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Catalogue coordinates must have shape (N, 3)")
    if not np.isfinite(positions).all() or not np.isfinite(box) or box <= 0:
        raise ValueError("Catalogue coordinates and a positive box size must be finite")
    if np.any(positions < 0) or np.any(positions >= box):
        raise ValueError("Prepared coordinates outside [0,L); refusing silent modulo wrap")
    if ids.shape != (len(positions),) or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("Unique nonnegative integer Subfind IDs must match catalogue rows")
    return np.ascontiguousarray(positions), box


def prepare_arrays(payload: dict[str, Any], with_provenance=False):
    """Generate or select D and generate R outside the timed region."""
    rng = np.random.default_rng(payload["seed"])
    n, box = int(payload["n"]), float(payload["box_size"])
    if payload.get("catalog"):
        catalog, box = load_catalog(payload["catalog"], box)
        if n > len(catalog):
            raise ValueError(f"Requested N={n}, but catalogue contains {len(catalog)} points")
        indices = rng.choice(len(catalog), size=n, replace=False)
        x = np.ascontiguousarray(catalog[indices])
        with np.load(payload["catalog"], allow_pickle=False) as data:
            sampled_ids = np.asarray(data["subfind_id"])[indices]
            snapshot = int(data["snapshot"])
        provenance = {"source_sha256": sha256_file(payload["catalog"]), "snapshot": snapshot,
                      "sample_indices_sha256": array_sha256(indices.astype("<i8")),
                      "sample_ids_sha256": array_sha256(sampled_ids.astype("<i8")),
                      "sampling": "without replacement; never augment undersized catalogues"}
    else:
        x = rng.uniform(0.0, box, size=(n, 3))
        provenance = {"source_sha256": None, "snapshot": None,
                      "sample_indices_sha256": None, "sample_ids_sha256": None,
                      "sampling": "explicit synthetic uniform generator"}
    y = rng.uniform(0.0, box, size=(n, 3))
    provenance.update({"coordinate_units": "cMpc/h", "data_array_sha256": array_sha256(x),
                       "random_array_sha256": array_sha256(y),
                       "duplicate_coordinate_excess": n-len(np.unique(x, axis=0))})
    if with_provenance:
        return x, y, box, provenance
    return x, y, box


def tree_histogram(tree_a: Any, tree_b: Any, edges: np.ndarray, auto: bool) -> np.ndarray:
    """Same raw unordered auto-count convention as spatial.pair_counts."""
    directed = np.asarray(tree_a.count_neighbors(tree_b, edges, cumulative=False), dtype=np.int64)[1:]
    if auto:
        if np.any(directed % 2):
            raise AssertionError("Positive-distance auto-pair count must be even")
        directed = directed // 2
    return directed


def dense_histogram(x: np.ndarray, y: np.ndarray | None, edges: np.ndarray, box: float) -> np.ndarray:
    """Optional full-matrix baseline, deliberately distinct from blocked brute."""
    auto = y is None
    other = x if auto else y
    delta = np.abs(x[:, None, :] - other[None, :, :])
    np.minimum(delta, box - delta, out=delta)
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    if auto:
        distances = distances[np.triu_indices(len(x), k=1)]
    else:
        distances = distances.ravel()
    bins = np.searchsorted(edges, distances, side="left") - 1
    valid = (bins >= 0) & (bins < len(edges) - 1)
    return np.bincount(bins[valid], minlength=len(edges) - 1).astype(np.int64)


def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    # Identical import footprint for ALL methods. In particular, importing
    # spatial also imports linalg/optimize/sparse: do not import it only in the
    # brute branch, or memory comparisons would reflect different libraries.
    import scipy.linalg
    import scipy.optimize
    import scipy.sparse
    import scipy.sparse.csgraph
    import scipy.spatial
    from scipy.spatial import cKDTree
    from spatial import brute_counts

    x, y, box, provenance = prepare_arrays(payload, with_provenance=True)
    edges = np.asarray(payload["edges"], dtype=np.float64)
    if edges.ndim != 1 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValueError("finite increasing nonnegative bin edges required")
    # Shared untimed validation above. No method-specific correctness warm-up.
    baseline = peak_rss_bytes()
    method = payload["method"]
    start = time.perf_counter()
    if method == "ckdtree":
        tree_d = cKDTree(x, boxsize=box)
        tree_r = cKDTree(y, boxsize=box)
        build_done = time.perf_counter()
        counts = {
            "DD": tree_histogram(tree_d, tree_d, edges, True),
            "DR": tree_histogram(tree_d, tree_r, edges, False),
            "RR": tree_histogram(tree_r, tree_r, edges, True),
        }
        done = time.perf_counter()
        build, query = build_done - start, done - build_done
    elif method == "blocked_brute":
        kwargs = {"block_size": payload["block_size"]} if "block_size" in inspect.signature(brute_counts).parameters else {}
        counts = {"DD": brute_counts(x, None, edges, box, **kwargs),
                  "DR": brute_counts(x, y, edges, box, **kwargs),
                  "RR": brute_counts(y, None, edges, box, **kwargs)}
        done = time.perf_counter()
        build, query = 0.0, done - start
    elif method == "dense_brute":
        counts = {"DD": dense_histogram(x, None, edges, box),
                  "DR": dense_histogram(x, y, edges, box),
                  "RR": dense_histogram(y, None, edges, box)}
        done = time.perf_counter()
        build, query = 0.0, done - start
    else:
        raise ValueError(f"Unknown method {method}")
    peak = peak_rss_bytes()
    serial_counts = {key: np.asarray(value, dtype=np.int64).tolist() for key, value in counts.items()}
    digest = hashlib.sha256(json.dumps(serial_counts, sort_keys=True).encode()).hexdigest()
    return {"status": "ok", "elapsed_total_s": done - start,
            "build_s": build, "query_s": query, "peak_rss_bytes": peak,
            "baseline_rss_bytes": baseline, "counts": serial_counts,
            "counts_sha256": digest, "box_size": box,
            "block_size_requested": payload["block_size"], **provenance}


def run_isolated(payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(payload)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "timed_out", "error": f"Worker exceeded {timeout:g} seconds and was terminated"}
    if proc.returncode:
        return {"status": "failed", "return_code": proc.returncode, "error": proc.stderr[-6000:]}
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "failed", "return_code": proc.returncode,
                "error": f"Worker did not emit valid JSON: {proc.stdout[-2000:]} {proc.stderr[-2000:]}"}
    result["return_code"] = proc.returncode
    return result


def summarize(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = []
    for n in sorted({row["n"] for row in rows}):
        for method in sorted({row["method"] for row in rows}):
            selected = [row for row in rows if row["n"] == n and row["method"] == method and row["status"] == "ok"]
            if not selected:
                continue
            item: dict[str, Any] = {"n": n, "method": method, "completed_repeats": len(selected)}
            for field in ("elapsed_total_s", "build_s", "query_s", "peak_rss_bytes", "baseline_rss_bytes"):
                values = [row[field] for row in selected]
                item[field] = {"median": statistics.median(values), "mean": statistics.mean(values),
                               "min": min(values), "max": max(values),
                               "sample_std": statistics.stdev(values) if len(values) > 1 else None}
            metrics.append(item)
    correctness = []
    for n, repeat in sorted({(row["n"], row["repeat"]) for row in rows}):
        selected = {row["method"]: row for row in rows if row["n"] == n and row["repeat"] == repeat and row["status"] == "ok"}
        tree = selected.get("ckdtree")
        if tree:
            for method, baseline in selected.items():
                if method == "ckdtree":
                    continue
                families = {key: tree["counts"][key] == baseline["counts"][key] for key in ("DD", "DR", "RR")}
                data_equal = all(tree.get(key) == baseline.get(key) for key in
                                 ("data_array_sha256", "random_array_sha256", "sample_indices_sha256", "source_sha256"))
                correctness.append({"n": n, "repeat": repeat, "reference": method,
                                    "family_equal": families, "same_input": data_equal,
                                    "exact_equal": data_equal and all(families.values())})
    speedups = []
    for item in metrics:
        if item["method"] != "ckdtree":
            continue
        for baseline in metrics:
            if baseline["n"] == item["n"] and baseline["method"] != "ckdtree":
                speedups.append({"n": item["n"], "baseline": baseline["method"],
                                 "baseline_over_tree_total": baseline["elapsed_total_s"]["median"] / item["elapsed_total_s"]["median"],
                                 "baseline_over_tree_peak_rss": baseline["peak_rss_bytes"]["median"] / item["peak_rss_bytes"]["median"]})
    failures = [dict((key, row[key]) for key in ("n", "repeat", "method", "status", "error") if key in row)
                for row in rows if row["status"] in ("failed", "timed_out")]
    tree_success = any(row["method"] == "ckdtree" and row["status"] == "ok" for row in rows)
    wrong = any(not test["exact_equal"] for test in correctness)
    status = "failed" if failures or wrong else "completed" if correctness and tree_success else "unverified"
    return {"metadata": metadata, "metrics": metrics, "correctness": correctness,
            "all_overlapping_counts_equal": bool(correctness) and all(test["exact_equal"] for test in correctness),
            "speedups": speedups, "worker_failures": failures,
            "skipped_runs": [{k:row[k] for k in ("n", "repeat", "method", "status", "error") if k in row}
                             for row in rows if row["status"].startswith("skipped")],
            "reference_status": "failed" if wrong else "passed" if correctness else "not_run",
            "status": status}


def write_outputs(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = ["origin", "catalog", "method", "n", "repeat", "seed", "status", "elapsed_total_s",
              "build_s", "query_s", "peak_rss_bytes", "baseline_rss_bytes", "box_size",
              "block_size_requested", "counts_sha256", "return_code", "error", "counts",
              "source_sha256", "snapshot", "coordinate_units", "sample_indices_sha256", "sample_ids_sha256",
              "data_array_sha256", "random_array_sha256", "duplicate_coordinate_excess", "sampling"]
    with (output / "benchmark_raw.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {**row, "counts": json.dumps(row.get("counts", {}), sort_keys=True)}
            writer.writerow(serialized)
    (output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-worker wall-clock cap in seconds")
    parser.add_argument("--brute-max", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--include-dense", action="store_true", help="Also run explicitly labelled full-matrix brute baseline")
    parser.add_argument("--dense-max", type=int, default=2000)
    parser.add_argument("--catalog", type=Path, help="Prepared, finite 3D positions NPZ; not downloaded automatically")
    parser.add_argument("--box-size", type=float, default=75.0, help="Synthetic box; real catalogue metadata is mandatory")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.worker:
        print(json.dumps(run_worker(json.loads(args.worker)), allow_nan=False))
        return 0
    if args.repeats < 1 or args.timeout <= 0 or args.box_size <= 0 or args.block_size < 1 or any(n < 2 for n in args.sizes):
        raise SystemExit("Positive timeout, box, block size, repeats and N>=2 are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Output directory must be new or empty; previous measured runs are preserved")
    catalog_size = None
    if args.catalog:
        args.catalog = args.catalog.resolve(strict=True)
        catalog_array, args.box_size = load_catalog(args.catalog, args.box_size)
        catalog_size = len(catalog_array)
        del catalog_array
    edges = np.geomspace(0.05, 5.0, 13)
    import scipy
    origin = "CATALOG_INPUT" if args.catalog else "SYNTHETIC_UNIFORM"
    metadata = {"origin": origin, "catalog": args.catalog.name if args.catalog else None,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "catalog_size": catalog_size, "source_sha256": sha256_file(args.catalog) if args.catalog else None,
                "coordinate_units": "cMpc/h", "python": sys.version,
                "platform": platform.platform(), "machine": platform.machine(),
                "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                "box_size": args.box_size, "edges": edges.tolist(),
                "bin_convention": "(left,right]", "sizes": args.sizes, "repeats": args.repeats,
                "brute_max": args.brute_max, "block_size": args.block_size,
                "timeout_seconds": args.timeout, "seed": args.seed,
                "random_catalog_size": "N_R=N_D for each worker",
                "timing_scope": "DD+DR+RR; tree total includes both tree builds; imports and data allocation excluded",
                "rss_scope": "Peak whole worker RSS including interpreter, imports and arrays, NOT incremental algorithm memory",
                "worker_import_policy": "All methods import spatial and scipy.linalg/optimize/sparse/sparse.csgraph/spatial before data preparation and timing",
                "validation_policy": "Shared untimed geometry/identity checks; no method-specific warm-up; inputs hashed per repeat",
                "baseline_policy": "blocked brute uses the public brute_counts routine, including its internal input checks; dense uses prevalidated arrays",
                "caveat": "Synthetic results do not establish galaxy-specific performance or physical conclusions"}
    methods = ["ckdtree", "blocked_brute"] + (["dense_brute"] if args.include_dense else [])
    rows: list[dict[str, Any]] = []
    for n in args.sizes:
        for repeat in range(args.repeats):
            seed = args.seed + 100003 * n + repeat
            for method in methods:
                row: dict[str, Any] = {"origin": origin, "catalog": args.catalog.name if args.catalog else "",
                                       "method": method, "n": n, "repeat": repeat + 1, "seed": seed}
                maximum = args.brute_max if method == "blocked_brute" else args.dense_max if method == "dense_brute" else None
                if (maximum is not None and n > maximum) or (catalog_size is not None and n > catalog_size):
                    row.update(status="skipped_size", error=f"N={n} exceeds method cap {maximum}" if maximum is not None and n > maximum else f"N={n} exceeds catalogue size {catalog_size}")
                else:
                    payload = {"method": method, "n": n, "seed": seed, "edges": edges.tolist(),
                               "box_size": args.box_size, "block_size": args.block_size,
                               "catalog": str(args.catalog) if args.catalog else None}
                    row.update(run_isolated(payload, args.timeout))
                rows.append(row)
                summary = summarize(rows, metadata)
                write_outputs(args.output_dir, rows, summary)
                print(f"{origin} N={n:>7} repeat={repeat + 1} {method:>13}: {row['status']}"
                      + (f" total={row['elapsed_total_s']:.6f}s RSS={row['peak_rss_bytes']/2**20:.2f}MiB" if row["status"] == "ok" else ""), flush=True)
    summary = summarize(rows, metadata)
    write_outputs(args.output_dir, rows, summary)
    print(f"Saved {args.output_dir.resolve()} | exact overlapping counts: {summary['all_overlapping_counts_equal']}", flush=True)
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
