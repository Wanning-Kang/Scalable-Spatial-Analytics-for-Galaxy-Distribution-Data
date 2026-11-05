"""One reproducible real-TNG execution; no synthetic fallback or key storage.

Run in a LOCAL terminal with --download --prompt-key for the first authenticated
download, or provide --raw-dir containing the five full-catalogue field files.
Every execution uses a new directory. Failures and missing-data stops are saved.
Heavy real-data runtime depends on selected N and periodic graph connectivity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from download_tng import FIELDS


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1048576), b""):
            value.update(part)
    return value.hexdigest()


def required_raw_files(raw_dir, cfg):
    return [Path(raw_dir)/cfg["simulation"]/f"snapshot_{snap:03d}"/f"{field}.hdf5"
            for snap in cfg["snapshots"] for field in FIELDS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"data/raw")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prompt-key", action="store_true", help="private input in a local terminal, not chat")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.prompt_key and not args.download:
        parser.error("--prompt-key requires --download")
    if args.prompt_key and not sys.stdin.isatty():
        parser.error("Private key prompting needs your local terminal; no key may be pasted into chat")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run-dir must be new or empty; old results are never replaced")
    raw_dir, output = args.raw_dir.resolve(), args.run_dir.resolve()
    args.config = args.config.resolve()
    if raw_dir == output or output in raw_dir.parents:
        parser.error("Raw input must not be inside this execution's output directory")
    output.mkdir(parents=True, exist_ok=True)
    state = {"mode": "REAL_TNG_ONLY", "started_utc": datetime.now(timezone.utc).isoformat(),
             "config": cfg, "config_sha256": digest(args.config), "raw_dir": str(raw_dir),
             "code_sha256": {p.name:digest(p) for p in sorted((ROOT/"src").glob("*.py"))},
             "status": "preflight", "stages": [],
             "warning": "No synthetic fallback. Completion means computations finished, not that a physical hypothesis was confirmed."}

    def save():
        (output/"execution.json").write_text(json.dumps(state, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    def stage(name, arguments, private_prompt=False):
        command = [sys.executable, str(ROOT/"src"/arguments[0]), *map(str, arguments[1:])]
        record = {"stage": name, "command": command, "started_utc": datetime.now(timezone.utc).isoformat()}
        state["stages"].append(record)
        state["status"] = "running"
        save()
        print(f"Starting {name}", flush=True)
        if private_prompt:
            # A getpass prompt is never captured to a log, and the key isn't in argv.
            result = subprocess.run(command, check=False)
        else:
            with (output/f"{name}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        record["return_code"] = result.returncode
        record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        if result.returncode:
            state["status"] = "stopped_stage_failure"
            save()
            return False
        save()
        return True

    missing = [str(p) for p in required_raw_files(raw_dir, cfg) if not p.is_file()]
    if missing and not args.download:
        state.update(status="blocked_missing_real_data", missing_files=missing)
        save()
        print("Real TNG data missing. Download in a local authenticated terminal, or specify --raw-dir. No science or synthetic substitute was run.")
        return 2
    if args.download:
        if not args.prompt_key and not os.environ.get("TNG_API_KEY"):
            state.update(status="blocked_missing_authentication")
            save()
            print("No TNG_API_KEY configured. Use --download --prompt-key in your local terminal; do not paste secrets into chat.")
            return 2
        if cfg["simulation"] != "TNG100-1" or set(cfg["snapshots"]) != {99, 67, 50, 33}:
            state.update(status="stopped_unsupported_download_configuration")
            save()
            return 2
        if not stage("download", ["download_tng.py", "--data-dir", raw_dir, *( ["--prompt-key"] if args.prompt_key else [])], args.prompt_key):
            return 2
    missing = [str(p) for p in required_raw_files(raw_dir, cfg) if not p.is_file()]
    if missing:
        state.update(status="blocked_incomplete_download", missing_files=missing)
        save()
        return 2
    prepared = output/"prepared"
    if not stage("prepare", ["prepare_tng.py", "--data-dir", raw_dir, "--output-dir", prepared,
                              "--mass-min-msun", cfg["stellar_mass_min_msun"],
                              "--stellar-particles-min", cfg["stellar_particle_min"],
                              "--snapshots", *cfg["snapshots"]]):
        return 2
    for snap in cfg["snapshots"]:
        if not stage(f"benchmark_{snap:03d}", ["benchmark.py", "--catalog", prepared/f"snapshot_{snap:03d}.npz",
                                               "--output-dir", output/f"benchmark_{snap:03d}", "--include-dense", "--repeats", 3]):
            return 2
    if not stage("analysis", ["run_analysis.py", "--config", args.config, "--prepared-dir", prepared,
                               "--output-dir", output/"analysis"]):
        return 2
    state.update(status="computations_completed_scientific_interpretation_required",
                 completed_utc=datetime.now(timezone.utc).isoformat(),
                 results="analysis/*/evidence_audit.json; no forced confirmation of expected CV finding")
    save()
    print(f"Computations completed. Review evidence_audit.json and controls before drawing scientific conclusions: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
