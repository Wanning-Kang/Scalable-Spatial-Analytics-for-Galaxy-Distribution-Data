"""Prepare REAL-DATA, ENGINEERING-ONLY 100k KD-tree benchmark coordinates.

This relaxed sample is NOT the physical main galaxy sample and MUST NOT be
used for galaxy-clustering or astrophysical conclusions. It only supplies an
adequately sized, real spatial catalog for computational scaling experiments.
Raw downloaded HDF5 files and their provenance are never modified.

Usage (the main agent/user executes this; importing this module writes nothing):
    python src/prepare_scaling_catalog.py --self-test
    python src/prepare_scaling_catalog.py --data-dir data/raw
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from download_tng import FIELDS, SIMULATION, atomic_json, sha256_file
from prepare_tng import read_field


SNAPSHOT = 99
BOX_CKPC_H = 75000.0
BOX_CMPC_H = 75.0
LITTLE_H = 0.6774
ENGINEERING_PARTICLE_MIN = 10
DEFAULT_N = 100000
DEFAULT_SEED = 20261021
DATA_KIND = "REAL_TNG100_1_ENGINEERING_ONLY_NOT_PHYSICAL_MAIN_SAMPLE"
WARNING = (
    "ENGINEERING-ONLY: relaxed stellar-particle/mass selection for algorithm "
    "scaling, NOT the >=1e9 Msun and >=100 stellar-particle physical main sample; "
    "do not use this catalog or its benchmark to infer galaxy clustering."
)


def id_sha256(ids):
    """Hash IDs in saved row order as contiguous little-endian signed int64."""
    canonical = np.ascontiguousarray(ids, dtype="<i8")
    if canonical.ndim != 1:
        raise ValueError("IDs must be one-dimensional.")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def select_engineering_candidates(arrays):
    """Return original catalog row IDs and an audit; never wrap coordinates.

    Eligibility: flag=1, finite positive stellar mass, >=10 stellar particles,
    finite coordinates in the strict half-open [0,75000) ckpc/h cube, and valid
    finite field values/counts/group IDs. IDs are original full-catalog row
    numbers, not newly numbered filtered rows. Distinct IDs sharing a coordinate
    cause an explicit stop rather than silent deduplication.
    """
    pos, mass, counts, flags, grnr = (np.asarray(arrays[field]) for field in FIELDS)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("SubhaloPos must have shape (N,3).")
    n = len(pos)
    shapes = {"SubhaloPos": (n, 3), "SubhaloMassType": (n, 6),
              "SubhaloLenType": (n, 6), "SubhaloFlag": (n,), "SubhaloGrNr": (n,)}
    for field in FIELDS:
        if np.asarray(arrays[field]).shape != shapes[field]:
            raise ValueError(f"Shape mismatch for {field}; catalog rows cannot be aligned safely.")
    stellar_mass = mass[:, 4].astype(np.float64) * 1e10 / LITTLE_H
    stellar_count = counts[:, 4]
    finite_position = np.isfinite(pos).all(axis=1)
    in_range = finite_position & (pos >= 0).all(axis=1) & (pos < BOX_CKPC_H).all(axis=1)
    finite_fields = (np.isfinite(mass).all(axis=1) & np.isfinite(counts).all(axis=1)
                     & np.isfinite(flags) & np.isfinite(grnr))
    valid_count = np.isfinite(stellar_count) & (stellar_count >= 0) & (stellar_count == np.floor(stellar_count))
    valid_halo = np.isfinite(grnr) & (grnr >= 0) & (grnr == np.floor(grnr))
    independently_eligible = (finite_fields & valid_count & valid_halo & (flags == 1)
                              & np.isfinite(stellar_mass) & (stellar_mass > 0)
                              & (stellar_count >= ENGINEERING_PARTICLE_MIN))
    candidate_ids = np.flatnonzero(independently_eligible & in_range).astype(np.int64)
    candidate_coords = pos[candidate_ids]
    if len(candidate_coords):
        _, first, repeated = np.unique(candidate_coords, axis=0, return_index=True, return_counts=True)
        duplicate_groups = int(np.count_nonzero(repeated > 1))
        if duplicate_groups:
            example_ids = candidate_ids[first[repeated > 1]][:10].tolist()
            raise ValueError(
                f"ENGINEERING-ONLY preparation stopped: {duplicate_groups} candidate coordinate "
                f"groups have distinct original IDs. Example first IDs: {example_ids}. "
                "No duplicates were dropped or jittered."
            )
    # Physical selection count is only a reference showing why this different
    # engineering selection is necessary; it does not relax the physical sample.
    physical = (finite_fields & valid_count & valid_halo & in_range & (flags == 1)
                & np.isfinite(stellar_mass) & (stellar_mass >= 1e9) & (stellar_count >= 100))
    finite_out_of_range = finite_position & ~in_range
    bounds_examples = []
    for index in np.flatnonzero(finite_out_of_range)[:20]:
        count = float(stellar_count[index])
        value = float(stellar_mass[index])
        bounds_examples.append({
            "subfind_id": int(index), "coords_ckpc_h": pos[index].astype(float).tolist(),
            "stellar_particle_count": count if np.isfinite(count) else None,
            "stellar_mass_msun": value if np.isfinite(value) else None,
            "engineering_eligible_without_geometry": bool(independently_eligible[index]),
        })
    audit = {
        "n_raw": n, "n_candidates": int(len(candidate_ids)),
        "candidate_ids_sha256": id_sha256(candidate_ids),
        "duplicate_candidate_coordinate_groups": 0,
        "n_reference_physical_main_sample": int(physical.sum()),
        "n_nonfinite_position_rows": int((~finite_position).sum()),
        "n_finite_out_of_range_rows": int(finite_out_of_range.sum()),
        "n_independently_eligible_but_invalid_geometry": int((independently_eligible & ~in_range).sum()),
        "out_of_range_examples": bounds_examples,
        "geometry_policy": "strict [0,75000) ckpc/h; no modulo wrap, clipping, jitter, duplication, or deduplication",
    }
    return candidate_ids, audit


def sample_without_replacement(candidate_ids, n=DEFAULT_N, seed=DEFAULT_SEED):
    """Sample unique original IDs reproducibly, then retain original row order."""
    ids = np.asarray(candidate_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("Candidate IDs must be one-dimensional nonnegative integers.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Candidate IDs must be distinct; never augment or repeat objects.")
    if int(n) != n or n < 1 or n > len(ids):
        raise ValueError(f"Cannot sample N={n} without replacement from {len(ids)} candidates.")
    chosen = np.random.default_rng(seed).choice(ids, size=int(n), replace=False)
    return np.sort(chosen.astype(np.int64))


def load_verified_snapshot(data_dir):
    """Load only snapshot099, requiring matching SHA256 and official-source manifests."""
    source = Path(data_dir) / SIMULATION / "snapshot_099"
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata_url = "https://www.tng-project.org/api/TNG100-1/snapshots/99/"
    if (metadata.get("simulation") != SIMULATION or metadata.get("snapshot") != SNAPSHOT
            or metadata.get("source_url") != expected_metadata_url):
        raise ValueError("Snapshot099 source metadata does not identify the expected official simulation/snapshot.")
    if (not np.isclose(float(metadata["boxsize_cMpc_h"]), BOX_CMPC_H)
            or not np.isclose(float(metadata["little_h"]), LITTLE_H)):
        raise ValueError("Source geometry/cosmology differs from the declared engineering selection.")
    redshift = float(metadata["redshift"])
    if not np.isfinite(redshift) or abs(redshift) > 1e-6:
        raise ValueError("Snapshot099 metadata must correspond to z approximately zero.")
    arrays, sources = {}, {}
    for field in FIELDS:
        path = source / f"{field}.hdf5"
        record_path = source / f"{field}.provenance.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        canonical_url = f"https://www.tng-project.org/api/{SIMULATION}/files/groupcat-99/?Subhalo={field}"
        before = path.stat()
        measured_sha = sha256_file(path)
        if (record.get("simulation") != SIMULATION or record.get("snapshot") != SNAPSHOT
                or record.get("field") != field or record.get("source_url") != canonical_url
                or record.get("sha256") != measured_sha
                or record.get("measured_bytes") != before.st_size):
            raise ValueError(f"Source/provenance SHA256 or identity mismatch for {field}.")
        arrays[field], header = read_field(path, field, SNAPSHOT)
        after = path.stat()
        if before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size:
            raise ValueError(f"Source changed during validation: {field}; retry without modifying raw files.")
        if "redshift" in header and not np.isclose(header["redshift"], redshift, rtol=0, atol=1e-7):
            raise ValueError(f"Redshift header mismatch for {field}.")
        sources[field] = {
            "path": path.relative_to(Path(data_dir)).as_posix(),
            "sha256": measured_sha, "measured_bytes": int(before.st_size),
            "provenance_path": record_path.relative_to(Path(data_dir)).as_posix(), "provenance_sha256": sha256_file(record_path),
            "retrieved_utc": record.get("retrieved_utc"), "validated_header": header,
        }
    if len(arrays["SubhaloPos"]) != int(metadata["num_subhalos"]):
        raise ValueError("Catalog length does not match the full official snapshot metadata.")
    safe_metadata = {key: metadata[key] for key in (
        "simulation", "snapshot", "redshift", "num_subhalos", "boxsize_cMpc_h",
        "little_h", "source_url", "retrieved_utc", "geometry_source") if key in metadata}
    return arrays, {"metadata": safe_metadata, "metadata_path": metadata_path.relative_to(Path(data_dir)).as_posix(),
                    "metadata_sha256": sha256_file(metadata_path), "fields": sources}


def prepare_scaling_catalog(data_dir, output_dir, n=DEFAULT_N, seed=DEFAULT_SEED,
                            expected_candidates=None, force=False):
    """Main write function; no file is written until all source/sample checks pass."""
    if n != DEFAULT_N:
        raise ValueError("This explicitly named 100k engineering artifact requires exactly N=100000.")
    destination = Path(output_dir) / "snapshot_099_scaling_100k.npz"
    provenance_path = Path(output_dir) / "snapshot_099_scaling_100k.engineering-only.provenance.json"
    if not force and (destination.exists() or provenance_path.exists()):
        raise ValueError("Engineering output already exists; pass --force explicitly to replace generated outputs only.")
    arrays, sources = load_verified_snapshot(data_dir)
    candidates, audit = select_engineering_candidates(arrays)
    if expected_candidates is not None and len(candidates) != int(expected_candidates):
        raise ValueError(f"Candidate count changed: got {len(candidates)}, expected {expected_candidates}.")
    ids = sample_without_replacement(candidates, n, seed)
    coords = arrays["SubhaloPos"][ids].astype(np.float64) / 1000.0
    if (not np.isfinite(coords).all() or np.any(coords < 0) or np.any(coords >= BOX_CMPC_H)
            or len(np.unique(coords, axis=0)) != n):
        raise ValueError("Engineering sample failed final finite/range/distinct-coordinate validation.")
    report = {
        "data_kind": DATA_KIND, "engineering_only": True, "warning": WARNING,
        "simulation": SIMULATION, "snapshot": SNAPSHOT,
        "sample_size": int(n), "seed": int(seed),
        "selection": {"SubhaloFlag": 1, "stellar_mass_msun": ">0 (finite)",
                      "stellar_mass_conversion": "SubhaloMassType[:,4] * 1e10 / 0.6774 Msun",
                      "stellar_particle_min": ENGINEERING_PARTICLE_MIN,
                      "position_interval_ckpc_h": "[0,75000) on every axis",
                      "other_validity": "finite mass/count/flag/group fields; integer nonnegative stellar counts and group IDs"},
        "physical_main_sample_unchanged": {"stellar_mass_min_msun": 1e9, "stellar_particle_min": 100,
                                           "note": "Separate physical main catalog; this engineering file must not replace it."},
        "audit": audit, "sources": sources,
        "sampling": "NumPy default_rng(seed).choice(original candidate IDs, replace=False); selected IDs sorted ascending",
        "bit_generator": "PCG64", "numpy_version": np.__version__,
        "sample_ids_sha256": id_sha256(ids),
        "id_hash_encoding": "contiguous little-endian signed int64 in saved ascending row order; raw bytes only",
        "coordinate_units": "cMpc/h", "coordinate_conversion": "raw ckpc/h divided by 1000; no h factor applied to comoving /h length",
        "boxsize_cMpc_h": BOX_CMPC_H, "periodic": True,
        "output_path": destination.relative_to(Path(output_dir)).as_posix(), "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="engineering-only.", suffix=".npz", dir=output_dir)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, coords=coords, coordinate_units=np.array("cMpc/h"),
                                boxsize_cMpc_h=np.float64(BOX_CMPC_H), snapshot=np.int64(SNAPSHOT),
                                subfind_id=ids, engineering_only=np.bool_(True),
                                data_kind=np.array(DATA_KIND), usage_warning=np.array(WARNING))
        report["output_sha256"] = sha256_file(Path(temporary))
        report["output_measured_bytes"] = int(Path(temporary).stat().st_size)
        os.replace(temporary, destination)
        atomic_json(provenance_path, report)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return report


def self_test():
    """Small in-memory helper tests only; never reads raw files or writes output."""
    pos = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                    [75000., 0., 0.], [12., 13., 14.], [15., 16., 17.]])
    mass = np.zeros((6, 6)); mass[:, 4] = .1; mass[4, 4] = 0
    counts = np.zeros((6, 6), dtype=int); counts[:, 4] = 10; counts[5, 4] = 9
    arrays = dict(zip(FIELDS, (pos, mass, counts, np.ones(6), np.arange(6))))
    ids, audit = select_engineering_candidates(arrays)
    np.testing.assert_array_equal(ids, [0, 1, 2])
    assert audit["n_candidates"] == 3 and audit["n_finite_out_of_range_rows"] == 1
    a = sample_without_replacement(ids, 2, DEFAULT_SEED)
    np.testing.assert_array_equal(a, sample_without_replacement(ids, 2, DEFAULT_SEED))
    assert len(np.unique(a)) == 2 and id_sha256(a) == id_sha256(a.astype(np.int32))
    for bad_ids, requested_n in ((np.array([0, 0, 1]), 2), (ids, 4)):
        try:
            sample_without_replacement(bad_ids, requested_n)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate IDs or oversized samples must stop.")
    arrays["SubhaloPos"] = pos.copy()
    arrays["SubhaloPos"][1] = pos[0]
    try:
        select_engineering_candidates(arrays)
    except ValueError as exc:
        assert "coordinate groups" in str(exc)
    else:
        raise AssertionError("Distinct IDs with duplicate coordinates must stop.")
    return {"status": "passed", "fixture": "in-memory synthetic helper only; not a real-data result"}


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=project / "benchmark_catalogues")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-candidates", type=int, default=None,
                        help="Optional independent count check, e.g. 106434 for the verified current snapshot099")
    parser.add_argument("--force", action="store_true", help="Replace generated engineering outputs only")
    parser.add_argument("--self-test", action="store_true", help="Only in-memory helper tests; no raw access/output writes")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
        return 0
    try:
        result = prepare_scaling_catalog(args.data_dir, args.output_dir, seed=args.seed,
                                         expected_candidates=args.expected_candidates, force=args.force)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ENGINEERING-ONLY preparation stopped: {exc}")
        return 1
    print(WARNING)
    print(f"Candidates: {result['audit']['n_candidates']}; sampled: {result['sample_size']}; seed: {args.seed}")
    print(f"Saved engineering scaling input: {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
