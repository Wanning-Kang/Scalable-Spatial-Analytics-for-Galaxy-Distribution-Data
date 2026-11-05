"""Small independent checks for the benchmark runner (python -m unittest)."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_right_closed_distinct_auto_pairs_and_periodic_distance(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [9., 0., 0.]])
        edges = np.array([0., 1., 2., 3.])
        # In a box of side 10: distances 1, 1, 2. Only unique pairs.
        np.testing.assert_array_equal(benchmark.dense_histogram(coords, None, edges, 10.), [2, 1, 0])

    def test_zero_distance_and_out_of_range_excluded(self):
        x = np.array([[0., 0., 0.]])
        y = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
        np.testing.assert_array_equal(benchmark.dense_histogram(x, y, np.array([0., 1., 2.]), 10.), [1, 1])

    def test_prepared_catalog_keys_and_determinism(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "catalog.npz"
            np.savez(path, coords=np.arange(36).reshape(12, 3) / 10., boxsize_cMpc_h=75.,
                     subfind_id=np.arange(12), snapshot=99, coordinate_units="cMpc/h")
            payload = dict(n=8, seed=6, box_size=10., catalog=str(path))
            first = benchmark.prepare_arrays(payload)
            second = benchmark.prepare_arrays(payload)
            self.assertEqual(first[2], 75.)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            self.assertEqual(len(np.unique(first[0], axis=0)), 8)
            provenance = benchmark.prepare_arrays(payload, with_provenance=True)[3]
            self.assertEqual(provenance["source_sha256"], benchmark.sha256_file(path))
            self.assertEqual(provenance["snapshot"], 99)
            with self.assertRaisesRegex(ValueError, "contains 12"):
                benchmark.prepare_arrays(dict(payload, n=13))

    def test_raw_units_missing_metadata_bounds_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "catalog.npz"
            base = dict(coords=np.zeros((3, 3)), boxsize_cMpc_h=75., subfind_id=np.arange(3), snapshot=99, coordinate_units="cMpc/h")
            for bad in [dict(base, coordinate_units="ckpc/h"), dict(base, coords=np.full((3,3), 75000.)),
                        dict(base, subfind_id=np.array([0,0,1])), {k:v for k,v in base.items() if k != "snapshot"}]:
                np.savez(path, **bad)
                with self.assertRaises(ValueError):
                    benchmark.load_catalog(path, 75.)

    def test_summary_does_not_verify_skipped_or_unreferenced_runs(self):
        summary = benchmark.summarize([dict(n=100000, repeat=1, method="ckdtree", status="skipped_size")], {})
        self.assertEqual(summary["status"], "unverified")
        self.assertEqual(summary["reference_status"], "not_run")
        self.assertFalse(summary["all_overlapping_counts_equal"])

    def test_summary_exact_equality_and_total_time_speedup(self):
        common = dict(n=5, repeat=1, status="ok", build_s=0., query_s=2., peak_rss_bytes=100,
                      baseline_rss_bytes=90, counts={"DD": [2], "DR": [3], "RR": [1]})
        rows = [dict(common, method="ckdtree", elapsed_total_s=2.),
                dict(common, method="blocked_brute", elapsed_total_s=6.)]
        summary = benchmark.summarize(rows, {"origin": "SYNTHETIC_UNIFORM"})
        self.assertTrue(summary["all_overlapping_counts_equal"])
        self.assertEqual(summary["speedups"][0]["baseline_over_tree_total"], 3.)
        rows[1]["counts"] = {"DD": [3], "DR": [3], "RR": [1]}
        summary = benchmark.summarize(rows, {})
        self.assertFalse(summary["all_overlapping_counts_equal"])
        self.assertEqual(summary["status"], "failed")

    def test_isolated_small_methods_match(self):
        common = dict(n=20, seed=20261009, edges=np.geomspace(.05, 5., 13).tolist(),
                      box_size=10., block_size=16, catalog=None)
        results = {method: benchmark.run_isolated(dict(common, method=method), 30.)
                   for method in ("ckdtree", "blocked_brute", "dense_brute")}
        for method, result in results.items():
            self.assertEqual(result["status"], "ok", f"{method}: {result.get('error')}")
            self.assertGreaterEqual(result["peak_rss_bytes"], result["baseline_rss_bytes"])
        self.assertEqual(results["ckdtree"]["counts"], results["blocked_brute"]["counts"])
        self.assertEqual(results["ckdtree"]["counts"], results["dense_brute"]["counts"])


if __name__ == "__main__":
    unittest.main()
