#!/usr/bin/env python3
"""Run real-catalogue smoke + four-snapshot benchmarks without overwriting runs.

Use the analysis virtualenv's Python to launch this driver. A completed smoke
test is required before formal runs. Every command gets an exclusive stdout
log and an attempt entry in the manifest; only validated completed directories
with matching source/code/arguments are resumed. Incomplete directories are
preserved and a numbered retry directory is used instead. No raw data changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def completed_summary(directory, expected):
    summary_path = directory / "benchmark_summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        meta = summary["metadata"]
        if (summary["status"] != "completed"
                or not summary["all_overlapping_counts_equal"]
                or summary.get("worker_failures")
                or meta.get("origin") != "CATALOG_INPUT"
                or meta.get("source_sha256") != expected["source_sha256"]
                or meta.get("sizes") != expected["sizes"]
                or meta.get("repeats") != expected["repeats"]
                or meta.get("seed") != expected["seed"]
                or meta.get("brute_max") != 5000
                or meta.get("block_size") != 512):
            return None
        tree = [item for item in summary["metrics"] if item["method"] == "ckdtree"]
        should_exist = [n for n in expected["sizes"] if n <= meta["catalog_size"]]
        if (sorted(item["n"] for item in tree) != sorted(should_exist)
                or any(item["completed_repeats"] != expected["repeats"] for item in tree)):
            return None
        return summary
    except (OSError, ValueError, TypeError, KeyError):
        return None


def output_directory(base):
    if not base.exists() or not any(base.iterdir()):
        return base
    for index in range(1, 10000):
        candidate = base.with_name(base.name + f"_retry{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many preserved attempts for {base}")


def run_one(name, snapshot, sizes, repeats, root, manifest, manifest_path, env, seed):
    benchmark = root / "src" / "benchmark.py"
    catalog = root / "prepared" / f"snapshot_{snapshot:03d}.npz"
    if not catalog.is_file():
        raise FileNotFoundError(f"Prepared real catalogue is required: {catalog}")
    expected = {"source_sha256": sha256_file(catalog), "benchmark_sha256": sha256_file(benchmark),
                "sizes": sizes, "repeats": repeats, "seed": seed,
                "snapshot": snapshot, "catalog": catalog.relative_to(root).as_posix()}
    for attempt in reversed(manifest["attempts"]):
        if (attempt["name"] == name and attempt.get("status") == "completed"
                and attempt.get("expected") == expected):
            summary = completed_summary(root / attempt["output_dir"], expected)
            if summary is not None:
                print(f"RESUME {name}: validated previous completed attempt {attempt['output_dir']}", flush=True)
                return True
    output = output_directory(root / "results" / name)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    log_path = root / "logs" / f"{name}_{identifier}.stdout.log"
    command = [sys.executable, str(benchmark), "--catalog", str(catalog),
               "--output-dir", str(output), "--sizes", *map(str, sizes),
               "--repeats", str(repeats), "--timeout", "90", "--brute-max", "5000",
               "--block-size", "512", "--include-dense", "--dense-max", "1000", "--seed", str(seed)]
    attempt = {"id": identifier, "name": name, "expected": expected,
               "output_dir": output.relative_to(root).as_posix(),
               "stdout_log": log_path.relative_to(root).as_posix(),
               "started_utc": utc_now(), "status": "running"}
    manifest["attempts"].append(attempt)
    manifest["updated_utc"] = utc_now()
    atomic_json(manifest_path, manifest)
    print(f"START {name}: catalogue={catalog.name}, sizes={sizes}", flush=True)
    try:
        with log_path.open("x", encoding="utf-8") as log:
            log.write(f"# started_utc: {attempt['started_utc']}\n# benchmark: {name}\n")
            log.flush()
            with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{name}] {line.rstrip()}", flush=True)
                returncode = proc.wait()
        summary = completed_summary(output, expected)
        attempt.update(return_code=returncode, finished_utc=utc_now(),
                       status="completed" if returncode == 0 and summary is not None else "failed")
        if summary is not None:
            attempt["summary"] = {"catalog_size": summary["metadata"]["catalog_size"],
                                  "actual_tree_sizes": [item["n"] for item in summary["metrics"] if item["method"] == "ckdtree"],
                                  "exact_comparisons": len(summary["correctness"]),
                                  "all_overlapping_counts_equal": summary["all_overlapping_counts_equal"],
                                  "skipped_runs": len(summary.get("skipped_runs", []))}
    except BaseException as error:
        attempt.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       finished_utc=utc_now(), error_type=type(error).__name__)
        manifest["updated_utc"] = utc_now()
        atomic_json(manifest_path, manifest)
        raise
    manifest["updated_utc"] = utc_now()
    atomic_json(manifest_path, manifest)
    print(f"END {name}: {attempt['status']}", flush=True)
    return attempt["status"] == "completed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--snapshots", type=int, nargs="+", default=[99, 67, 50, 33])
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    for directory in (root / "results", root / "logs", root / ".matplotlib"):
        directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # The benchmark subprocess does not need the private TNG download key.
    env.pop("TNG_API_KEY", None)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLCONFIGDIR=str(root / ".matplotlib"))

    # Record only the dependencies needed by the benchmark, never pip freeze
    # (which may contain local editable paths or private index/VCS URLs).
    packages = {}
    for name in ("numpy", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    environment = {"python_version": platform.python_version(),
                   "platform": platform.platform(), "packages": packages,
                   "overrides": {"OPENBLAS_NUM_THREADS": env["OPENBLAS_NUM_THREADS"],
                                 "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
                                 "MPLCONFIGDIR": ".matplotlib"}}
    path = root / "results" / "benchmark_driver_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("attempts"), list):
            raise ValueError("Unexpected existing manifest; not overwriting it")
    else:
        manifest = {"schema_version": 1, "created_utc": utc_now(), "attempts": [], "invocations": []}
    manifest["invocations"].append({"started_utc": utc_now(), "driver_sha256": sha256_file(__file__),
                                     "environment": environment, "snapshots_requested": args.snapshots})
    atomic_json(path, manifest)
    if not run_one("benchmark_099_smoke", 99, [500, 1000, 5000], 1, root, manifest, path, env, args.seed):
        print("Smoke failed: formal benchmark runs were not launched.", flush=True)
        return 2
    for snapshot in args.snapshots:
        if not run_one(f"benchmark_{snapshot:03d}", snapshot,
                       [500, 1000, 5000, 10000, 50000, 100000], 3,
                       root, manifest, path, env, args.seed):
            print("A formal benchmark failed: remaining runs were not launched.", flush=True)
            return 2
    manifest["invocations"][-1].update(finished_utc=utc_now(), status="completed")
    atomic_json(path, manifest)
    print(f"Completed real-catalogue benchmarks; manifest: {path.relative_to(root)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
