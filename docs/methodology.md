# Methodology

This document describes the **TNG100-1 simulation-catalogue analysis** reported in this repository. It is not an observational galaxy survey, and its local labels and selected subsets do not establish a physical causal mechanism.

## Study design and input catalogue

The main analysis uses snapshots **99, 67, 50, 33** (approximately **z = 0, 0.5, 1, 2**) from one periodic TNG100-1 simulation box. The box length is **75 cMpc/h** in each direction. The same eligibility rules apply across snapshots: `SubhaloFlag = 1`, total bound stellar mass >= 10^9 solar masses, and `SubhaloLenType[:, 4] >= 100` as a type-4 resolution proxy. There are **21,052**, **19,621**, **17,836**, and **12,281** selected galaxy-hosting subhalos, respectively. See [Data access](data_access.md) for source fields, units, and audit qualifications.

Coordinates are converted from `ckpc/h` to `cMpc/h` by dividing by 1,000. They are neither standardized to z-scores nor divided by `h` a second time. The original simulation's periodic geometry is retained. For a box of length `L`, component-wise minimum-image displacement is

$$
\Delta_i = \min(|x_i-y_i|,\; L-|x_i-y_i|),\qquad
r = \sqrt{\sum_i \Delta_i^2}.
$$

`Subfind ID` identifies a record **within** a snapshot; it is not a persistent cross-snapshot galaxy ID. Quality checks reject invalid selected positions, missing fields/units and duplicate selected IDs, rather than silently repairing them.

## Pair counting and two-point statistics

All distances use the same fixed 12-bin edge array specified in `config.json`, spanning **0.05–5 cMpc/h**. The implementation counts unordered distinct pairs for `DD` and `RR`, and cross-pairs for `DR`. Zero-distance pairs are excluded, and the bin convention is **(left, right]**. The periodic `scipy.spatial.cKDTree` implementation is compared against blocked brute-force pair counting on reference-size samples; large catalogues are not asserted to have an independent brute-force reference.

With normalized pair counts

$$
dd=DD/\binom{N_D}{2},\quad dr=DR/(N_D N_R),\quad rr=RR/\binom{N_R}{2},
$$

the code evaluates five estimators:

| Estimator | Expression |
|---|---|
| Natural | `dd/rr - 1` |
| Davis–Peebles | `dd/dr - 1` |
| Hewett | `(dd - dr)/rr` |
| Hamilton | `dd*rr/(dr*dr) - 1` |
| Landy–Szalay | `(dd - 2*dr + rr)/rr` |

Landy–Szalay is the designated primary reported correlation function. Undefined denominator bins are represented as missing/NaN, not manufactured zero estimates. The **scientific** analysis uses a random catalogue with `N_R = 5 N_D`; the **performance benchmarks** use `N_R = N_D` and therefore do not have identical workload sizes.

## Predeclared scales, fitting and uncertainty

The *small-scale* summary is the **unweighted mean of the three distance bins wholly inside the predeclared [0.2, 1] cMpc/h interval**; together they span approximately **0.232–0.734 cMpc/h**. It is not an integral over the full continuous interval. Power-law parameters summarize a shell-volume-averaged model over six wholly included bins in the predeclared [0.2, 3] interval, covering approximately **0.232–2.321 cMpc/h**. Fitting uses covariance-aware nonlinear least squares with recorded regularization and diagnostics. Numerical convergence is **not** evidence that a single power law fits adequately; see [Results](results.md).

Internal spatial uncertainty is estimated with a **3 x 3 x 3 = 27-region delete-one jackknife**. In each replication, data and random points in the same spatial region are removed while keeping the original periodic box length. Correlated comparisons reuse the same random catalogue where appropriate. The jackknife describes internal variation, **not** independent cosmic-variance uncertainty. When subsets or removal samples are compared, labels determined on the complete original sample remain fixed; a label-reidentification uncertainty is not included.

## Local environment and approximate grouping

Local galaxy **number-density proxy**, not stellar mass density, is defined through the distance to the **k-th other neighbor**:

$$
\rho_k = k / \left[(4\pi/3)\,d_k^3\right].
$$

The main analysis uses `k = 20`, with `k = 10` as a sensitivity check. Objects are ranked and divided into **high 20% / medium 60% / low 20%** density environments. These groups are coordinate-derived subsets rather than independent physically randomized treatments.

HDBSCAN uses periodic distances. Small catalogues can use an exact dense periodic-distance matrix; full catalogues use a **periodic sparse-neighbor-graph approximation** with graph connectivity and neighborhood checks. Main settings include `min_cluster_size = 40`, `min_samples = 10`, starting `graph_k = 32`, and a configurable upper bound of 256. Sensitivity comparisons cover graph neighborhood size and `min_cluster_size` values **20, 40, 80**. Cluster membership strength, hierarchy persistence, and compactness have different meanings; no one quantity should be renamed an independently calibrated physical cluster probability.

## Four global–local analyses

1. **Density-conditioned correlation:** estimate and compare correlations in high-, medium- and low-density subsets.
2. **Cluster versus noise:** compare group-assigned and HDBSCAN noise-labelled samples; noise is a clustering label, not bad input data.
3. **Targeted removal versus random removal:** remove the highest-density 5%, 10%, or 20% and compare with equal-number random removals (10 random repeats in the reported removal control).
4. **Raw `DD` composition:** partition observed data-data pairs into within-group, between-group, cluster–noise, and noise–noise categories.

Counts are recomputed and normalized for each subset with its actual `N_D`. The raw `DD` categories sum to total observed `DD` where defined, but **their fractions are not additive contributions to Landy–Szalay excess correlation**, which also depends on `DR`, `RR`, and normalization. Density selection and removal are descriptive conditional analyses, not interventions that isolate astrophysical causality.

## Additional controls and cross-redshift comparisons

Each original snapshot analysis includes a same-size **uniform synthetic control** and a **coarse stellar-mass-histogram-matched** sample. The four snapshots share a simulation volume and initial conditions: they are **not four independent universes**. Mass matching equalizes counts in specified coarse stellar-mass bins and total sample sizes; it does not remove every selection, resolution, or within-bin mass difference.

An unusual negative original `z≈2` uniform-control statistic motivated **20 additional independently seeded D/R realizations per catalogue size**, retained *alongside* the original rather than replacing it. A separate short-distance pair-count check supports the stored `DD/DR/RR` arithmetic for that case; it does not fully calibrate the null distribution or the jackknife. See [Limitations](limitations.md).

## Implementation boundaries

Principal modules include `src/prepare_tng.py`, `src/spatial.py`, `src/feature_audit.py`, `src/run_analysis.py`, `src/run_staged.py`, and `src/evidence.py`. Benchmark methodology is documented separately in [Benchmarking](benchmarking.md). The values cited here reflect the **saved 2026-10-21 experiment and its later diagnostic review**, not an independently rerun result from this published repository copy.
