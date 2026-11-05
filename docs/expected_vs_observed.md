# Expected versus observed: recorded differences

The historical project plan mixed **proposed methods**, **expected qualitative patterns**, and some **draft conclusions**. This page compares those expectations with the **saved TNG100-1 run of 2026-10-21** and its **2026-10-22 result-review notes**. Numerical measurements below were not known in the original plan merely because the plan listed sample sizes and thresholds. “Supported” means consistent within this study, **not** independent causal or universal physical proof.

| Original expectation or question | Recorded observation | Assessment |
|---|---|---|
| Periodic cKDTree and brute force yield the same reference-size DD/DR/RR counts. | Exact integer histogram agreement in tested overlapping sizes; historical record includes 60 method comparisons and 180 histograms across four snapshots. | **Supported** for tested sizes, not extrapolated to 100K. |
| Very small brute-force problems might be just as fast or faster. | At the smallest tested **N=500**, cKDTree was already ~3.83–3.94x faster. Below 500 was not measured. | **Not observed within the tested range.** |
| Indexing scales faster, with better memory behavior than brute force. | At N=5,000 cKDTree was ~24.93–25.33x faster than blocked brute force. Dense full-matrix memory grew markedly, but block-limited brute force did **not** show the same N² resident-memory growth. | **Speed supported; blanket memory claim qualified.** |
| A scientific sample can be benchmarked to 100,000 objects. | Strict scientific samples contain **12,281–21,052** objects. A distinct **100,000-object broader engineering sample** was created from Snapshot 99; no 100K brute-force reference was run. | **Engineering scaling supported; not a 100K strict scientific sample.** |
| Galaxy correlations are stronger at small separations and decrease with distance. | Strong positive small-scale signals were measured. At the farthest tested separation bin the correlation remained positive; zero was **not** established within 5 cMpc/h. | **Supported trend with range restriction.** |
| Low-redshift samples have stronger small-scale clustering; trend persists after some mass matching. | Predeclared main-sample means at z≈0/0.5/1/2: **141.81 / 87.41 / 64.74 / 45.82**. Coarse mass-matched means: **135.76 / 84.44 / 60.77 / 45.82**. | **Supported descriptively**; same volume and incomplete control of confounders. |
| A simple power law will fit intermediate scales well. | Optimization converged, but reduced chi-squared **2.905–5.470** and residual structure indicate an inadequate universal single-law summary. | **Not supported as a good-fit claim.** |
| The fitted correlation length `r0` changes monotonically with redshift. | Values at z≈1 and z≈2 are close, with z≈2 slightly larger; the original proposal did not explicitly require strict monotonicity. | **Do not invent this as a falsified original hypothesis.** |
| Dense groups are stable and have greater membership strengths. | Group density correlates positively with **compactness** and **persistence** but negatively with **mean membership strength** (four snapshots). | **Metric-dependent; blanket stability claim contradicted.** |
| High density should cluster most strongly; medium should exceed low density. | High is largest in the predefined mean for each snapshot. At **z≈2**, low exceeds medium on the point estimate; uncertainty limits interpretation of the reversal. | **High strongest supported; strict medium/low ordering not supported.** |
| Cluster-labelled objects correlate more strongly than noise. | Cluster–noise mean ξ differences were positive for all four snapshots. | **Supported as coordinate-conditioned comparison**, not physical causal proof. |
| Targeted high-density removal should reduce small-scale correlation more than random removal. | Top-10%-density deletion reduced mean ξ by **31.7–62.8%**; equal-count random deletion mean changes were much smaller. Farthest bin also changed by about 25–28%. | **Supported, with noncausal and far-bin caveats.** |
| The tiniest-separation raw pair counts are mostly within the same cluster. | Same-cluster raw `DD` fraction was **59.2–71.2%** in the smallest bin; **noise–noise** still made up about **28.6–40.0%**. | **Supported for raw counts**, not as Landy–Szalay signal attribution. |
| Uniform synthetic controls should be centered near zero. | Original z≈2 control was **-0.322 ± 0.081 JK SE**; added **20 independent D/R realizations per size**, keeping the original, left some calibration concerns unresolved. | **Not fully validated.** |

## Further qualifications

- **Member strength versus persistence:** these describe different properties of the HDBSCAN hierarchy; the latter's positive association with group density does not reverse the former's negative association.
- **Physical groups:** the clustering algorithm's groups should not be treated as individually validated FoF haloes. Dominant FoF membership fractions in recorded groups are moderate or low, especially at higher z.
- **Pair composition:** in the three-bin small-scale aggregate, same-cluster pairs contributed **72.2–89.6% of raw DD**. That statistic is not the contribution to normalized excess correlation and is not causal allocation.
- **Unusual uniform control:** the independent short-distance DD/DR/RR check reproduced saved pair arithmetic for the z≈2 control, but did not resolve its tail probability, uncertainty calibration, or cause. The extra 20-run experiment was a **subsequent diagnostic**, not a replacement of the original or an anticipated design step.
- **No direct study:** multiple new stellar-mass thresholds, DBSCAN-versus-HDBSCAN, a Hamilton before/after normalization comparison, truth labels for all noise subtypes, and full cosmic-variance uncertainty were **not** directly tested. They should not be presented as findings.

## Revised defensible conclusion

For four consistently selected TNG100-1 snapshots, the saved experiment supports **small-scale spatial clustering**, stronger conditional correlation for **coordinate-defined high-density environments**, larger small-scale changes following **targeted high-density removal** than random removal, and a majority of **raw** close `DD` pairs within the same assigned groups. The relation is **descriptive**, and the observations do not independently prove that compact groups physically *cause* the measured excess-correlation signal. At the same time, a single power law is only a rough fit; medium/low density rankings, membership-strength interpretations and uniform-control calibration require care.

See [Results](results.md) for full quantitative tables and [Limitations](limitations.md) for the statistical boundaries.
