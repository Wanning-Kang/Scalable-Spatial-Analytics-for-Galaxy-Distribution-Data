# Scalable-Spatial-Analytics-for-Galaxy-Distribution-Data
Spatial analysis of 3D galaxy distributions using scalable indexing, density estimation, clustering, and two-point correlation statistics.

A Python workflow for studying three-dimensional galaxy distributions in IllustrisTNG, combining periodic spatial indexing, two-point correlation statistics, local density estimation, and density-based clustering.

## Project overview

This project connects computational scalability with descriptive global–local clustering analysis.

The workflow includes:

- Quality filtering, coordinate-unit conversion, and spatial features.
- Periodic cKDTree pair counting with brute-force reference checks.
- Five two-point correlation estimators.
- kNN density estimation and periodic sparse-graph HDBSCAN.
- Power-law fitting and diagnostics.
- Spatial jackknife uncertainty estimation.
- Density-conditioned correlations, targeted-removal controls, and raw pair-count decomposition.

## Data

The scientific analysis uses TNG100-1 snapshots 99, 67, 50, and 33.

| Snapshot | Approximate redshift | Selected objects |
|----------|----------------------|-----------------:|
| 99       | 0                    | 21,052           |
| 67       | 0.5                  | 19,621           |
| 50       | 1                    | 17,836           |
| 33       | 2                    | 12,281           |

Coordinates are expressed in cMpc/h in a periodic box of side length 75 cMpc/h.

The scientific selection uses `SubhaloFlag = 1`, total bound stellar mass of at least 10^9 solar masses, and `SubhaloLenType[:, 4] >= 100` as a resolution proxy.

Raw TNG data and API credentials are not included. See [data access](docs/data_access.md).

## Selected results

- Checked DD, DR, and RR counts matched brute-force references on 500–5,000-point samples.
- At 5,000 points, periodic cKDTree pair counting was approximately **25 times faster** than the blocked brute-force implementation on the tested machine.
- Scaling to **100,000 objects** was measured using a separate, more broadly selected engineering catalogue.
- Removing the highest-density 10% reduced the predefined small-scale mean correlation by **31.7–62.8%** across the four snapshots.

These are measurements for the stated samples, configuration, implementation, and hardware—not universal performance guarantees or independent causal conclusions.

![Engineering scaling benchmark](figures/benchmark_100k.png)

![Targeted-removal comparison](figures/density_removal.png)

See [full results](docs/results.md) and [benchmark methodology](docs/benchmarking.md).

## Installation

The original experiment was run with Python 3.12 on macOS. Other platforms must be validated separately.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Run offline tests

```bash
python -m pytest tests -q
```

Tests use small artificial fixtures and do not require a TNG API key.

## Run a synthetic demonstration

```bash
python src/run_analysis.py \
  --demo \
  --output-dir outputs/demo_01
```

This demonstration uses artificial data. Its outputs are not TNG scientific findings.

Use a new output directory for another run.

## Reproduce the TNG analysis

Download the specified catalogue fields using your own TNG account:

```bash
python src/download_tng.py \
  --data-dir data/raw \
  --prompt-key
```

Enter the API key only at the hidden prompt. Do not put it in source code or commit it.

Prepare the scientific catalogues:

```bash
python src/prepare_tng.py \
  --data-dir data/raw \
  --output-dir prepared
```

Run the staged scientific analysis:

```bash
python src/run_staged.py \
  --prepared-dir prepared \
  --output-dir outputs/science_01
```

See [reproducibility instructions](docs/reproducibility.md) for performance benchmarks, the engineering-only 100K catalogue, additional diagnostics, and figure generation.

## Limitations

- The 100K engineering catalogue is not the scientific main sample.
- Full-catalogue HDBSCAN uses a periodic sparse-graph approximation.
- Spatial jackknife describes internal spatial uncertainty; it is not a complete estimate of cosmic variance.
- Subset and removal resamples retain labels from the full sample.
- A single power law does not adequately describe all fitted curves.
- Raw DD fractions are not fractions of Landy–Szalay excess correlation.
- Uniform-null diagnostics retain deviations whose calibration is not fully resolved.
- Density selection and removal results do not establish independent physical causation.

See [limitations](docs/limitations.md) and [expected versus observed results](docs/expected_vs_observed.md).

## Repository contents

- `src/`: analysis and supporting tools.
- `tests/`: offline tests.
- `docs/`: methods, data access, results, and limitations.
- `figures/`: selected final figures.
- `results/`: lightweight numerical evidence.
- `.github/workflows/`: automated test configuration.

## Data attribution and licensing

This project uses publicly released IllustrisTNG simulation data.

See the [official IllustrisTNG citation guidance](https://www.tng-project.org/data/docs/background/) and the references listed in [data access](docs/data_access.md).

Any license applied to the repository's original code does not replace the conditions applicable to TNG data or third-party software.
