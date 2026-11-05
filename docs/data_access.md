# Data access, units and provenance

## Source and scope

This project analyzes **publicly released IllustrisTNG TNG100-1 simulation catalogues**. It is not an observational survey and the selected points are not simulated by this repository. The scientific analysis reads four snapshot identifiers: **99, 67, 50, and 33**.

Official documentation:

- [IllustrisTNG data specifications](https://www.tng-project.org/data/docs/specifications/)
- [IllustrisTNG background and citation guidance](https://www.tng-project.org/data/docs/background/)

Consult the official release documentation for the relevant scientific publications and data-use/citation conditions. This repository's original-code license, if any, does **not** supersede the conditions for TNG data or third-party packages.

## Downloaded fields

The original workflow requests these **five** subhalo catalogue fields:

| Field | Purpose |
|---|---|
| `SubhaloPos` | Three-dimensional subhalo position |
| `SubhaloMassType` | Bound mass by particle type; type-4 component is used for the stellar-mass selection |
| `SubhaloLenType` | Particle counts by type; type-4 component acts as a resolution proxy |
| `SubhaloFlag` | Selection flag |
| `SubhaloGrNr` | FoF-halo index for limited catalogue-based grouping comparisons |

In the historical study, the source selection was `SubhaloFlag == 1`, `SubhaloMassType[:,4] * 1e10 / h >= 1e9 Msun`, and `SubhaloLenType[:,4] >= 100`, with **h = 0.6774**. The type-4 count is a **resolution proxy**, not an independently verified per-particle census of formed stars versus wind-phase objects. The box side length is **75 cMpc/h**. Raw positions use `ckpc/h`, converted to `cMpc/h` through division by 1,000; no further division by `h` is applied to positions.

## Sample counts

| Snapshot | Redshift (historical header/metadata) | Raw subhalo catalogue rows | Selected objects |
|---|---:|---:|---:|
| 99 | ~0 | 4,371,211 | 21,052 |
| 67 | ~0.503 | 4,765,222 | 19,621 |
| 50 | ~0.997 | 5,127,294 | 17,836 |
| 33 | ~2.002 | 5,621,338 | 12,281 |

These are results for the stated selection, not a claim of catalogue completeness under an external survey selection function. No selected duplicate coordinates or IDs were found in the recorded main samples. An out-of-box position in a **rejected raw record** was logged rather than silently wrapped; selected out-of-box records would stop preprocessing. Exact box-length coordinates may be mapped to periodic zero as a boundary equivalence.

## Retrieval and preparation

From the repository root, with your own TNG access credentials:

```bash
python src/download_tng.py --data-dir data/raw --prompt-key
python src/prepare_tng.py --data-dir data/raw --output-dir prepared
```

Enter the API key only through the hidden terminal prompt. **Never** place a live API key in committed code, documentation, screenshots, logs, `.env` files, or example commands. The repository does **not** include raw downloaded TNG HDF5 data, prepared per-object NPZ catalogues, or API credentials.

The original download workflow recorded API source identity, response metadata, byte counts, and local SHA-256 hashes. These are useful for *local integrity and provenance checks*, but a locally generated manifest is **not** an independent checksum published by the upstream TNG team. Some field-subset HDF5 files do not independently provide snapshot identity through an internal snapshot header; identity also depends on the requested API path and stored retrieval record.

Preprocessing should reject missing or inconsistent sources and units, record selection counts, and retain a quality report. Only commit **reviewed, non-identifying, small** public summary files. See [Reproducibility](reproducibility.md) for local run requirements.

## Separate 100K engineering input

The 100,000-object scalability benchmark does **not** use the strict scientific sample. The historical study used a **separately selected engineering-only** Snapshot 99 catalogue: `SubhaloFlag == 1`, positive stellar mass, at least 10 type-4 particles, and legal positions. From **106,434** eligible candidates, **100,000 real TNG objects** were sampled without replacement. No artificial duplicate or jittered galaxies were added to fill the count. The wider selection changes what the benchmark represents; it must not be used as a 100K-object high-mass scientific sample. See [Benchmarking](benchmarking.md).

## Public-release boundary

Keep raw HDF5, per-object coordinates, private provenance manifests, downloaded file trees, account tokens, generated logs, and environment records on the local machine. Review JSON, CSV, images, and Markdown for local absolute paths and metadata before publishing. A `.gitignore` is **not** a substitute for checking files manually uploaded through GitHub's web interface. This repository uses relative paths for examples and documentation.
