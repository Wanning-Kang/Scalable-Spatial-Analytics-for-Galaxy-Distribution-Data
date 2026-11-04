"""Validate and select TNG100-1 galaxies from five field-only HDF5 files.

Selection assumptions: SubhaloFlag=1, total stellar mass >=1e9 Msun,
stellar particle count >=100. These are declared project choices rather than
official completeness claims. Coordinates remain comoving, in cMpc/h.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from download_tng import FIELDS, SIMULATION, SNAPSHOTS, atomic_json, sha256_file

BOX_SIZE = 75.0
LITTLE_H = 0.6774
MASS_MIN_MSUN = 1e9
STAR_PARTICLES_MIN = 100


def read_field(path: Path, field: str, snapshot: int) -> tuple[np.ndarray, dict]:
    header = {}
    with h5py.File(path, "r") as handle:
        if f"Subhalo/{field}" in handle:
            values = np.asarray(handle[f"Subhalo/{field}"])
        elif field in handle:
            values = np.asarray(handle[field])
        else:
            raise ValueError(f"Required dataset {field} is missing from {path.name}.")
        attrs = dict(handle.attrs)
        if "Header" in handle:
            attrs.update(dict(handle["Header"].attrs))
        for name in ("SnapshotNumber", "SnapNum", "snapshot", "snapshot_number", "Number"):
            if name in attrs:
                actual = int(np.asarray(attrs[name]).item())
                if actual != snapshot:
                    raise ValueError(f"Snapshot header mismatch in {path.name}: expected {snapshot}, got {actual}.")
                header["snapshot"] = actual
        for name in ("Redshift", "redshift"):
            if name in attrs:
                header["redshift"] = float(np.asarray(attrs[name]).item())
        for name in ("BoxSize", "boxsize"):
            if name in attrs:
                box_ckpc_h = float(np.asarray(attrs[name]).item())
                if not np.isclose(box_ckpc_h / 1000.0, BOX_SIZE, rtol=1e-6, atol=1e-8):
                    raise ValueError(f"BoxSize mismatch in {path.name}; expected 75000 ckpc/h.")
                header["boxsize_ckpc_h"] = box_ckpc_h
    return values, header


def prepare_snapshot(data_dir: Path, output_dir: Path, snapshot: int,
                     mass_min_msun: float = MASS_MIN_MSUN,
                     stellar_particles_min: int = STAR_PARTICLES_MIN) -> dict:
    directory = data_dir / SIMULATION / f"snapshot_{snapshot:03d}"
    source_metadata = {}
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if source_metadata.get("snapshot") != snapshot or source_metadata.get("simulation") != SIMULATION:
            raise ValueError("Source metadata simulation/snapshot does not match the requested source.")
    arrays, headers, provenance = {}, {}, {}
    for field in FIELDS:
        path = directory / f"{field}.hdf5"
        arrays[field], headers[field] = read_field(path, field, snapshot)
        measured_sha = sha256_file(path)
        manifest_path = directory / f"{field}.provenance.json"
        verification = "sha256 calculated; download provenance unavailable"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("simulation") != SIMULATION or manifest.get("snapshot") != snapshot
                    or manifest.get("field") != field or manifest.get("sha256") != measured_sha
                    or manifest.get("measured_bytes") != path.stat().st_size):
                raise ValueError(f"Source provenance mismatch for {field}.")
            verification = "download provenance and sha256 verified"
        provenance[field] = {"path": path.relative_to(data_dir).as_posix(), "sha256": measured_sha,
                             "measured_bytes": path.stat().st_size, "verification": verification}
    pos, mass, counts, flags, grnr = (arrays[field] for field in FIELDS)
    if pos.ndim != 2:
        raise ValueError("SubhaloPos must be a two-dimensional full-catalog coordinate array.")
    n_raw = len(pos)
    expected_count = source_metadata.get("num_subhalos")
    if expected_count is not None and int(expected_count) != n_raw:
        raise ValueError("Catalog row count does not match the source snapshot metadata.")
    expected_shapes = {"SubhaloPos": (n_raw, 3), "SubhaloMassType": (n_raw, 6),
                       "SubhaloLenType": (n_raw, 6), "SubhaloFlag": (n_raw,), "SubhaloGrNr": (n_raw,)}
    for field in FIELDS:
        if arrays[field].shape != expected_shapes[field]:
            raise ValueError(f"Shape mismatch for {field}: {arrays[field].shape}, expected {expected_shapes[field]}.")
    redshifts = [header["redshift"] for header in headers.values() if "redshift" in header]
    if "redshift" in source_metadata:
        redshifts.append(float(source_metadata["redshift"]))
    if redshifts and not np.allclose(redshifts, redshifts[0], rtol=0, atol=1e-7):
        raise ValueError("Redshift metadata is inconsistent across the source fields.")
    # Invalid nonfinite coordinates are excluded and audited. Finite out-of-box
    # coordinates are a geometry error and must not be repaired by modulo wrap.
    finite_pos = np.isfinite(pos).all(axis=1)
    pos64 = pos.astype(np.float64)
    invalid_bounds = finite_pos & ((pos64 < 0).any(axis=1) | (pos64 > 75000.0).any(axis=1))
    wrapped_exact_boundary_components = int(np.count_nonzero(pos64 == 75000.0))
    pos64[pos64 == 75000.0] = 0.0
    coords = pos64 / 1000.0
    stellar_mass = mass[:, 4].astype(np.float64) * 1e10 / LITTLE_H
    stellar_count = counts[:, 4]
    finite_fields = (finite_pos & np.isfinite(mass).all(axis=1)
                     & np.isfinite(counts).all(axis=1) & np.isfinite(flags) & np.isfinite(grnr))
    valid_integer_counts = (np.isfinite(stellar_count) & (stellar_count >= 0)
                            & (stellar_count == np.floor(stellar_count)))
    valid_halo_ids = (np.isfinite(grnr) & (grnr >= 0) & (grnr == np.floor(grnr)))
    mask = (finite_fields & valid_integer_counts & valid_halo_ids & (flags == 1)
            & (stellar_mass >= mass_min_msun) & (stellar_count >= stellar_particles_min))
    # Audit the raw catalogue, but do not let an independently ineligible
    # subhalo prevent analysis of valid selected galaxies. Never modulo-wrap
    # an out-of-bounds candidate or silently exclude an otherwise eligible one.
    if np.any(invalid_bounds & mask):
        raise ValueError(f"Coordinates outside [0,75000] ckpc/h for {int((invalid_bounds & mask).sum())} selection-eligible records.")
    bounds_examples = [
        {"subfind_id": int(i), "coords_ckpc_h": pos[i].astype(float).tolist(),
         "stellar_mass_msun": float(stellar_mass[i]),
         "stellar_particle_count": int(stellar_count[i]),
         "selection_eligible": bool(mask[i])}
        for i in np.flatnonzero(invalid_bounds)[:20]
    ]
    selected = np.flatnonzero(mask)
    selected_coords = coords[mask]
    if len(selected_coords):
        _, duplicate_counts = np.unique(selected_coords, axis=0, return_counts=True)
        duplicate_excess = int(np.maximum(duplicate_counts - 1, 0).sum())
        duplicate_objects = int(duplicate_counts[duplicate_counts > 1].sum())
    else:
        duplicate_excess = duplicate_objects = 0
    redshift = redshifts[0] if redshifts else float("nan")
    quality = {
        "simulation": SIMULATION, "snapshot": snapshot,
        "redshift": None if not np.isfinite(redshift) else redshift,
        "n_raw": n_raw, "n_selected": len(selected),
        "selection": {"SubhaloFlag": 1, "stellar_mass_min_msun": mass_min_msun,
                      "stellar_particle_min": stellar_particles_min,
                      "note": "Total bound stellar mass and >=100 star particles are project assumptions; no completeness claim."},
        "coordinates": {"input_units": "ckpc/h", "output_units": "cMpc/h", "boxsize": BOX_SIZE,
                        "little_h": LITTLE_H, "periodic": True,
                        "exact_L_components_mapped_to_zero": wrapped_exact_boundary_components},
        "exclusions_nonexclusive": {"nonfinite_fields": int((~finite_fields).sum()),
                                   "invalid_counts": int((~valid_integer_counts).sum()),
                                   "invalid_halo_ids": int((~valid_halo_ids).sum()),
                                   "flag_not_one": int((flags != 1).sum()),
                                   "stellar_mass_below_threshold": int((stellar_mass < mass_min_msun).sum()),
                                   "star_particles_below_threshold": int((stellar_count < stellar_particles_min).sum())},
        "duplicate_selected_coordinates": {"excess_records": duplicate_excess,
                                           "objects_in_duplicate_groups": duplicate_objects,
                                           "action": "retained: distinct subfind IDs may share coordinates"},
        "raw_out_of_bounds_coordinates": {
            "count": int(invalid_bounds.sum()), "selected_count": int((invalid_bounds & mask).sum()),
            "examples_first_20": bounds_examples,
            "action": "not wrapped; independently ineligible objects remain excluded; eligible out-of-bounds objects stop preparation"},
        "source_headers": headers, "source_fields": provenance,
        "snapshot_validation": "headers/provenance checked where available; files lacking snapshot headers cannot be independently matched by content",
        "subfind_id_note": "Original full-catalog row index, preserved before selection; do not compare IDs across snapshots as persistent galaxy IDs.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"snapshot_{snapshot:03d}.npz", coords=selected_coords,
                        subfind_id=selected.astype(np.int64), stellar_mass_msun=stellar_mass[mask],
                        stellar_particle_count=stellar_count[mask].astype(np.int64),
                        halo_id=grnr[mask].astype(np.int64), snapshot=np.int64(snapshot),
                        redshift=np.float64(redshift), boxsize_cMpc_h=np.float64(BOX_SIZE),
                        little_h=np.float64(LITTLE_H), coordinate_units=np.array("cMpc/h"))
    atomic_json(output_dir / f"snapshot_{snapshot:03d}.quality.json", quality)
    return quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshots", nargs="+", type=int, default=list(SNAPSHOTS))
    parser.add_argument("--mass-min-msun", type=float, default=MASS_MIN_MSUN)
    parser.add_argument("--stellar-particles-min", type=int, default=STAR_PARTICLES_MIN)
    args = parser.parse_args()
    if not set(args.snapshots).issubset(SNAPSHOTS):
        parser.error("Supported experiment snapshots are 99, 67, 50, and 33.")
    if args.mass_min_msun <= 0 or args.stellar_particles_min < 1:
        parser.error("Mass and particle-count thresholds must be positive.")
    try:
        for snapshot in args.snapshots:
            quality = prepare_snapshot(args.data_dir, args.output_dir, snapshot,
                                       args.mass_min_msun, args.stellar_particles_min)
            print(f"Prepared snapshot {snapshot}: {quality['n_selected']}/{quality['n_raw']} selected", flush=True)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Preparation stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
