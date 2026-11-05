"""Miniature HDF5 tests for selection, units, identity, and source integrity."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import download_tng
from download_tng import FIELDS, DownloadError, download_field, sha256_file
from prepare_tng import prepare_snapshot


class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw" / "TNG100-1" / "snapshot_099"
        self.source.mkdir(parents=True)
        # Rows 0 and 5 have identical coordinates but distinct subfind IDs.
        pos = np.array([[1000, 2000, 3000], [100, 200, 300], [400, 500, 600],
                        [700, 800, 900], [np.nan, 0, 0], [1000, 2000, 3000],
                        [75000, 0, 0]], dtype=np.float32)
        mass = np.zeros((7, 6), dtype=np.float32)
        mass[:, 4] = .1  # 1.476e9 Msun after h conversion
        mass[2, 4] = .01
        counts = np.zeros((7, 6), dtype=np.int32)
        counts[:, 4] = 120
        counts[3, 4] = 99
        flags = np.array([1, 0, 1, 1, 1, 1, 1], dtype=np.int16)
        grnr = np.array([7, 8, 9, 10, 11, 12, 13], dtype=np.int32)
        for field, values in zip(FIELDS, (pos, mass, counts, flags, grnr)):
            self.write_field(field, values)
        (self.source / "metadata.json").write_text(json.dumps({
            "snapshot": 99, "simulation": "TNG100-1", "redshift": 0.0}), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def write_field(self, field, values, snapshot=99):
        with h5py.File(self.source / f"{field}.hdf5", "w") as handle:
            header = handle.create_group("Header")
            header.attrs["SnapshotNumber"] = snapshot
            header.attrs["Redshift"] = 0.0
            header.attrs["BoxSize"] = 75000.0
            handle.create_group("Subhalo").create_dataset(field, data=values)

    def test_selection_units_ids_duplicates_and_boundary(self):
        audit = prepare_snapshot(self.root / "raw", self.root / "prepared", 99)
        data = np.load(self.root / "prepared" / "snapshot_099.npz")
        np.testing.assert_array_equal(data["subfind_id"], [0, 5, 6])
        np.testing.assert_array_equal(data["halo_id"], [7, 12, 13])
        np.testing.assert_allclose(data["coords"], [[1, 2, 3], [1, 2, 3], [0, 0, 0]])
        np.testing.assert_allclose(data["stellar_mass_msun"], .1 * 1e10 / .6774, rtol=1e-6)
        self.assertEqual(str(data["coordinate_units"]), "cMpc/h")
        self.assertEqual(audit["n_selected"], 3)
        self.assertEqual(audit["duplicate_selected_coordinates"]["excess_records"], 1)
        self.assertEqual(audit["coordinates"]["exact_L_components_mapped_to_zero"], 1)

    def test_header_snapshot_mismatch_rejected(self):
        self.write_field("SubhaloFlag", np.ones(7, dtype=np.int16), snapshot=33)
        with self.assertRaisesRegex(ValueError, "Snapshot header mismatch"):
            prepare_snapshot(self.root / "raw", self.root / "prepared", 99)

    def test_source_manifest_mismatch_rejected(self):
        (self.source / "SubhaloFlag.provenance.json").write_text(json.dumps({
            "snapshot": 33, "simulation": "TNG100-1", "field": "SubhaloFlag",
            "sha256": sha256_file(self.source / "SubhaloFlag.hdf5"),
            "measured_bytes": (self.source / "SubhaloFlag.hdf5").stat().st_size}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            prepare_snapshot(self.root / "raw", self.root / "prepared", 99)

    def test_out_of_bounds_not_silently_wrapped(self):
        pos = np.zeros((7, 3), dtype=np.float64)
        pos[0, 0] = 75001.0
        self.write_field("SubhaloPos", pos)
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_snapshot(self.root / "raw", self.root / "prepared", 99)

    def test_out_of_bounds_independently_ineligible_is_audited_not_wrapped(self):
        pos = np.zeros((7, 3), dtype=np.float64)
        pos[2, 1] = 75000.0390625  # Row 2 fails the mass cut independently.
        self.write_field("SubhaloPos", pos)
        audit = prepare_snapshot(self.root / "raw", self.root / "prepared", 99)
        bounds = audit["raw_out_of_bounds_coordinates"]
        self.assertEqual(bounds["count"], 1)
        self.assertEqual(bounds["selected_count"], 0)
        self.assertEqual(bounds["examples_first_20"][0]["coords_ckpc_h"][1], 75000.0390625)
        with np.load(self.root / "prepared" / "snapshot_099.npz") as data:
            self.assertNotIn(2, data["subfind_id"])

    def test_shape_mismatch_rejected(self):
        self.write_field("SubhaloMassType", np.zeros((7, 5)))
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            prepare_snapshot(self.root / "raw", self.root / "prepared", 99)

    def test_metadata_row_count_mismatch_rejected(self):
        (self.source / "metadata.json").write_text(json.dumps({
            "snapshot": 99, "simulation": "TNG100-1", "redshift": 0.0,
            "num_subhalos": 8}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "row count"):
            prepare_snapshot(self.root / "raw", self.root / "prepared", 99)


class TestDownloader(unittest.TestCase):
    """Mocked offline downloads; no account key or live request is involved."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_key_attempts_no_download(self):
        with patch.dict(os.environ, {"TNG_API_KEY": ""}), \
                patch.object(sys, "argv", ["download_tng.py", "--data-dir", str(self.root / "raw")]), \
                patch.object(download_tng, "open_authenticated") as request, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(download_tng.main(), 2)
            request.assert_not_called()
        self.assertFalse((self.root / "raw").exists())

    def test_mock_download_has_measured_hash_and_reuses_verified_file(self):
        fixture = self.root / "fixture.hdf5"
        with h5py.File(fixture, "w") as handle:
            handle.create_dataset("SubhaloFlag", data=np.array([1, 0], dtype=np.int16))
        payload = fixture.read_bytes()
        with patch.object(download_tng, "open_authenticated", return_value=io.BytesIO(payload)), \
                contextlib.redirect_stdout(io.StringIO()):
            manifest = download_field(self.root / "raw", 99, "SubhaloFlag", "offline-fixture-key")
        self.assertEqual(manifest["measured_bytes"], len(payload))
        target = self.root / "raw" / "TNG100-1" / "snapshot_099" / "SubhaloFlag.hdf5"
        self.assertEqual(manifest["sha256"], sha256_file(target))
        self.assertNotIn("offline-fixture-key", json.dumps(manifest))
        with patch.object(download_tng, "open_authenticated") as request, \
                contextlib.redirect_stdout(io.StringIO()):
            download_field(self.root / "raw", 99, "SubhaloFlag", "offline-fixture-key")
            request.assert_not_called()

    def test_existing_file_without_provenance_is_not_overwritten(self):
        directory = self.root / "raw" / "TNG100-1" / "snapshot_099"
        directory.mkdir(parents=True)
        target = directory / "SubhaloFlag.hdf5"
        target.write_bytes(b"existing file retained")
        with patch.object(download_tng, "open_authenticated") as request:
            with self.assertRaisesRegex(DownloadError, "--force"):
                download_field(self.root / "raw", 99, "SubhaloFlag", "offline-fixture-key")
            request.assert_not_called()
        self.assertEqual(target.read_bytes(), b"existing file retained")

    def test_invalid_response_does_not_leave_partial_file(self):
        with patch.object(download_tng, "open_authenticated", return_value=io.BytesIO(b"invalid response")):
            with self.assertRaisesRegex(DownloadError, "not an HDF5"):
                download_field(self.root / "raw", 99, "SubhaloFlag", "offline-fixture-key")
        directory = self.root / "raw" / "TNG100-1" / "snapshot_099"
        self.assertEqual(list(directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
