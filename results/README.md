# Published numerical summaries

This directory intentionally contains **small, reviewed summaries**, not raw IllustrisTNG particle/subhalo catalogues, per-object positions, complete run logs, locally generated original outputs, or machine-specific file paths.

## Contents

- [`science_summary_public.json`](science_summary_public.json): four main TNG100-1 snapshots, fitted clustering descriptors, and coarse stellar-mass-matched comparisons.
- [`benchmark_summary_public.json`](benchmark_summary_public.json): four main-catalogue benchmark series and one separate **real TNG, broader-eligibility 100K engineering** scaling series.
- [`uniform_null_summary_public.json`](uniform_null_summary_public.json): 20 seeded uniform-null realizations per catalogue size and the retained anomalous Snapshot 033 control.

## Interpretation and provenance

These are selected statistics derived from saved historical experiment outputs. The *release preparation* performed consistency checks; it **did not rerun the complete TNG100-1 scientific experiment**. Values may be rounded to eight decimals for readability. For full methodology, see [`../docs/methodology.md`](../docs/methodology.md), [`../docs/benchmarking.md`](../docs/benchmarking.md), and [`../docs/limitations.md`](../docs/limitations.md).

- The small-scale statistic is the equal-weight mean of three complete bins inside the declared 0.2–1 cMpc/h range, **not** a continuous integral.
- Spatial jackknife error describes internal spatial variation across 27 regions, not independent cosmic variance.
- At N=5,000 the tree/brute comparison demonstrates exact agreement of overlapping DD, DR and RR counts and roughly 25x faster DD+DR+RR wall time for the tested implementation. No 100K brute-force comparison was run.
- The 100K engineering catalogue contains actual TNG objects drawn under broader selection criteria; it is **not** the main scientific sample and is **not** a synthetic point cloud.
- The original Snapshot 033 uniform-null negative result is retained as an unresolved calibration warning. The additional 20 outcomes are post-hoc diagnostics, not a calibrated significance test.
- Removing high-density objects and comparing raw pair shares do not establish astrophysical causation.

The public summaries deliberately omit large covariance matrices, intermediate per-experiment data, coordinates and original private-path metadata. Reproducing all figures or numerical analyses from these three summary JSONs **alone is not possible**; see `../docs/reproducibility.md` for the full-data workflow and access requirements.
