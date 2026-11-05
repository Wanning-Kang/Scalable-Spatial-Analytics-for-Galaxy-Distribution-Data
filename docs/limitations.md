# Limitations and interpretation boundaries

The purpose of this page is to state what the **saved TNG100-1 experiment** supports, and what it does **not** establish. Measurements here describe a chosen simulation catalogue, selection function, implementation and internal uncertainty convention. This is not an independent astrophysical causal inference study.

## Simulation and sample selection

- The four snapshots (**99, 67, 50, 33**) are from the **same simulation volume**; they are not independent cosmological volumes. Cross-redshift comparisons have shared initial conditions.
- `SubhaloFlag`, stellar-mass threshold and type-4 particle count are **project-defined** sample restrictions. `SubhaloLenType[:,4]` is a **resolution proxy**, not a particle-by-particle validation of formed stellar content.
- Mass matching equalizes **coarse** stellar-mass histogram counts, not every physical covariate or bin-internal property. No full stellar-mass-threshold sensitivity sweep was performed.
- Per-snapshot `Subfind ID` is not a persistent object-tracking identifier across the four snapshots.

## Two-point statistics and fitting

- The correlation remains **positive** in the furthest tested separation bin (3.406–5 cMpc/h); the measurement does not show that it has reached zero within the plotted distance range.
- The five estimators were calculated on the same counts. Selecting Landy–Szalay as primary does **not** demonstrate that it has universally the smallest variance or bias on every bin; a definitive all-estimator ranking was not made.
- A single shell-averaged power law has **reduced chi-squared ~2.9–5.47** and noticeable residuals. Successful optimization and non-boundary estimates do not constitute a good-fit certificate, and a comprehensive alternative-model comparison was not completed.
- The inferred `r0` values are not strictly monotone between z≈1 and z≈2. Do not rewrite this as a clean monotone sequence.
- The summary called “small-scale 0.2–1” uses only three complete bins spanning about **0.232–0.734 cMpc/h**; it is not a continuous-range integral or a claim about all sub-bin separations.

## Density conditioning, HDBSCAN and physical interpretation

- The local quantity `rho_k = k/V_k` is a coordinate-based **number-density proxy**, not stellar-mass density or an independently measured physical environment.
- Full-scale HDBSCAN uses a **periodic sparse-kNN-graph approximation**. A connected graph and high small-sample dense/sparse adjusted Rand index do not prove that its minimum spanning tree or all memberships equal the full dense-distance solution.
- Changing `min_cluster_size` changes labels nontrivially (reported ARI **0.604–0.811** versus the default in the sensitivity comparisons). “Perfectly parameter-stable clustering” is unsupported.
- Cluster group compactness, hierarchical persistence and mean membership strength are distinct. The study found **positive** relationships with the first two and **negative** relationships with mean membership strength across snapshots. Membership strength is not a calibrated probability that a point is in a physical galaxy cluster.
- HDBSCAN **noise** does not mean corrupt data. Proposed physical categories such as isolated galaxies, filament bridges or cluster outskirts were not individually truth-labelled or exhaustively validated.
- Dominant FoF-halo fractions for computed groups were not high enough to equate all density groups with verified physical galaxy clusters.

## Conditional contrasts are not causal allocation

- High-density objects have stronger conditional small-scale correlation in the saved sample, but the density partitions are built from the **same positions** used to estimate correlation. They are not independent treatments.
- In the z≈2 case, the **medium-versus-low density** small-scale point estimates reverse order; the paired uncertainty does not establish a statistically significant reverse physical effect. No strict all-snapshot, all-scale ordering is claimed.
- Targeted 10% deletion reduced the predefined small-scale mean ξ by **31.7–62.8%**; that does not isolate a unique physical cause of clustering. Random deletion controls are useful but do not make this a causal experiment.
- In deletion and subset resampling, the original sample's labels are **held fixed**; the estimate omits uncertainty from relabelling under each resample.
- Pair-composition fractions describe **raw observed DD**, not normalized Landy–Szalay excess correlation. They do not decompose the correlation function into additive “percent caused by each class.”

## Uncertainty and uniform controls

- The 27-region spatial delete-one jackknife measures **internal spatial variability** in a fixed simulation box, not complete **cosmic variance** (particularly modes larger than the box). Its normal-interval interpretation is approximate and can be affected by boundaries and shared cross-region pairs.
- The random-removal “± SD” records variability between **10 random-deletion repeats**, not the internal jackknife or independent universes.
- The original **z≈2 uniform control mean ξ = -0.322 ± 0.081 (internal JK SE)** deviates materially from zero. Twenty added uniform realizations retained the original result, but do **not** provide a reliable extreme-tail calibration. Some additional mean intervals also exclude zero under approximate assumptions. The cause of residual calibration deviations is **not fully resolved**.
- Independent short-distance DD/DR/RR arithmetic checks reproduce saved counts for the anomalous control. That verifies a **limited counting calculation**, not the null distribution, jackknife covariance or astrophysical model.
- Additional null realizations and diagnostics were **post-hoc** responses to the observed anomaly and must be described accordingly.

## Benchmark and reproducibility scope

- Exact cKDTree/brute-force DD/DR/RR comparisons were only run where the implementation caps allowed them (**500–5,000 objects**); larger sizes do not have brute-force equality evidence.
- The separate 100K sample uses **broader engineering-only selection**, not the strict scientific main sample, and **no 100K brute-force timing** was measured. Reported memory is the **whole worker peak RSS**, not solely pair-counting algorithm memory.
- Historical timings and numerical outputs are from a specified macOS/Python environment. A clean installation of the current release copy on other machines has not been certified by these documents; dependency versions and platform-specific RSS reporting should be rechecked.
- The source of published numbers is the **saved original experiment documentation**. Source review and small synthetic tests are not a re-execution of the complete TNG analysis. Scientific replication requires the original raw fields, parameter configuration, code version, source integrity checks, and a fresh run.
- An apparent title/axis mismatch exists on the provided **cross-redshift mass-matched figure**: the title says “stellar-mass histograms” whereas axes show correlation vs separation. Regenerate with a corrected title before calling that illustration final.

## Tests not performed in the recorded study

The study did **not** directly complete a full changed-stellar-mass-threshold survey, a matched DBSCAN-versus-HDBSCAN comparison, a before/after Hamilton normalization bug experiment, physical truth-labelling of all local structures, or a full ensemble of independent cosmological volumes. No result from those absent studies is inferred here.

For the measured quantities and the retained anomalous controls, see [Results](results.md). For a structured account of original hypotheses and actual findings, see [Expected versus observed](expected_vs_observed.md).
