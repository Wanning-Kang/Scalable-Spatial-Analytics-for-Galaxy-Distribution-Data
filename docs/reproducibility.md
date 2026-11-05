# Reproducibility and release checklist

This page distinguishes **historical recorded TNG results**, **offline code tests**, and a **fresh reproducible run**. The public repository contains source code and selected figures, but **not the raw TNG catalogues or complete private run history**. The current release copy has not been certified by an independent full four-snapshot rerun.

## 1. Environment and prerequisites

The historical experiment was conducted on **macOS using Python 3.12**. A user should begin from the repository root in a clean Python environment and validate dependency availability rather than assume the historical measurements are portable:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

`requirements-dev.txt` refers to `requirements.txt`. The original pinned requirements already included `pytest`; direct **Pillow** availability is still a release-integration check because updated figure-verification tooling imports it. Confirm the final dependency lists against the **published** versions of all modules before promising one-command installation.

The tests use small synthetic or mocked fixtures and should not require API credentials. A saved-fixture-dependent test may skip if its private results are absent. A historical local run reported **62 passed, 1 skipped**, while an earlier release-copy review did **not** execute one HDBSCAN-dependent test; neither statement proves every test on the **current** GitHub checkout passes. Do not publish an unconditional “all tests passed” claim without running the suite in a fresh installed environment.

## 2. Fetch official catalogue fields

Obtain your own authorized TNG account and use the public download script:

```bash
python src/download_tng.py --data-dir data/raw --prompt-key
```

Provide your key **only through the hidden prompt**. This step requests the five selected group-catalogue fields for snapshots **99, 67, 50, 33** and records local provenance and SHA-256. Never commit or paste actual credentials, download URLs containing credentials, `.env` values, shell histories, or private download logs. Local provenance manifests confirm recorded download integrity; they are **not** upstream independent checksum attestations.

## 3. Prepare the scientific input catalogue

```bash
python src/prepare_tng.py \
  --data-dir data/raw \
  --output-dir prepared
```

The historical configuration is in `config.json`; it includes: simulation `TNG100-1`; snapshots `[99,67,50,33]`; `boxsize_cMpc_h=75`; stellar mass threshold `1e9 Msun`; type-4 particle count threshold 100; edges spanning **0.05–5 cMpc/h**; `random_ratio=5`; `jackknife_grid=3`; `knn_k=20`; `min_cluster_size=40`; `min_samples=10`; sparse neighbor-graph settings and declared density-removal controls. Prepared catalogues are locally generated and are **not published** as per-object point clouds.

Check the quality reports for selected-object counts, validity of units, box boundaries, identity and input hashes. The four original selected counts were **21,052 / 19,621 / 17,836 / 12,281** (99 / 67 / 50 / 33). A different input release, seed, selection or code version may produce different results; document any discrepancy rather than overwriting the recorded original numbers.

## 4. Run an offline demonstration first

```bash
python src/run_analysis.py --demo --output-dir outputs/demo_01
```

`--demo` uses **synthetic points**, not real TNG galaxy data, and is a basic environment/process sanity check only. Use a **new output directory** for each trial, never reuse one containing a partial or completed recorded run.

## 5. Run the full scientific cases locally

```bash
python src/run_staged.py \
  --prepared-dir prepared \
  --config config.json \
  --output-dir outputs/science_01
```

This is the staged entry point for twelve analyses: four real main catalogues, four same-size synthetic uniform controls, and four coarse mass-matched cases. Its source and input fingerprints are intended to guard resumption; **a changed code/config/input requires a new output directory**. It preserves partial cases and does not silently replace them. Data paths, interpreter paths and environment details in **newly generated local run records** should be treated as private until explicitly sanitized. The public `docs/` must never include raw run-state or whole-machine environment dumps.

The 2026-10-21 scientific report records `N_R = 5N_D`, 27 spatial jackknife regions, the five normalized correlation estimators, periodic kNN/HDBSCAN, removal controls and separate null diagnostics. Merely running `--demo` or the tests does **not** reproduce those recorded numbers.

## 6. Benchmark and engineering scaling, separately

For a small offline benchmark smoke test:

```bash
python src/benchmark.py --sizes 500 1000 --repeats 1 \
  --output-dir outputs/benchmark_smoke
```

For the real prepared inputs, the dedicated driver is:

```bash
python src/run_benchmarks.py --root .
```

The historical benchmark used equal-size **data and random** samples; the scientific run used five times as many random points. In the formal runs, block-limited brute-force baselines were capped at **5,000** and full-matrix brute force at **1,000**. The driver may write environment metadata and logs: **do not publish these unreviewed**.

The **100K engineering catalogue** is generated *separately* with a broader, documented sample definition. The preparer accepts explicit `--data-dir` and `--output-dir` arguments and has an in-memory helper check:

```bash
python src/prepare_scaling_catalog.py --self-test
python src/prepare_scaling_catalog.py \
  --data-dir data/raw \
  --output-dir benchmark_catalogues
```

The full run also requires the real downloaded fields and local provenance manifests. It must not be represented as a 100K-object high-mass science sample or as a verified 100K tree/brute comparison.

## 7. Figures, checks and published evidence

The six selected PNGs under `figures/` are **presentation copies**, not full numerical replication inputs:

- `benchmark_100k.png`
- `cross_redshift_mass_matched.png`
- `density_subsets.png`
- `density_removal.png`
- `pair_composition.png`
- `uniform_null_ensemble.png`

The original figure workflows and local result verification may require a full original run tree, private per-case outputs, and supplementary diagnostic results. The release-edited figure verifier performs **automated file/format checks**, not proof of complete visual or scientific correctness. There is a **known cross-redshift figure title/axis mismatch** (title refers to stellar-mass histograms, axes show correlation versus separation) that should be corrected through regeneration before final public-facing endorsement.

If publishing small scientific summaries, use an **explicit allowlist** and a `results/public_manifest.json` with hashes of **the actual released copies**, not hashes of private originals with different content. Some experiments have historical source manifests and evidence audits, but they are not automatically safe to publish. Do not upload original `integrity_manifest` trees or raw point arrays. Before publishing, check that exact source, sample, configuration, code and measurements correspond to every claimed number.

## 8. Interpretation and attribution

The reported small-scale statistic is an equal-weight mean of three bins wholly inside 0.2–1 cMpc/h, not an integral over the full interval. `jackknife` error is **internal spatial uncertainty**, not total cosmic variance. The 20 new uniform controls added after a surprising z≈2 null result are **post-hoc** checks; the original negative result remains in the record. Neither high-density removal nor raw DD composition establishes independent astrophysical causality. See [Methodology](methodology.md), [Results](results.md), and [Limitations](limitations.md).

Consult the [IllustrisTNG citation guidance](https://www.tng-project.org/data/docs/background/) before scientific reuse or publication. The source data are third-party simulation releases, and a repository code license does not relicense them.
