# RT Computation Status Audit

Generated: 2026-06-26 13:37 UTC (updated 2026-06-28 18:12 UTC — MMA people upload confirmed complete)

Counts JSON files in `assets/results/<RT>/` and compares against the
expected combinatorial total from configuration constants.
Expected counts are derived from: models=['sd1.4'], tasks=['breeds', 'scenes', 'people'],
methods=['distil', 'munba', 'uce'], |Me|=51, |Mp|=5, |s|=5.

| RT | Actual | Expected | Status | Notes |
|-------|--------|----------|--------|-------|
| CountSignificantRelationship (CSR) | 3 | 3 | ✅ DONE | 1×3 |
| MetricMetricAlignment (MMA) | 11382 local | 11475 | ✅ DONE | people/munba=1282, people/uce=1282, people/distil=1282. Root cause of original gap: DNS failure + CRLF in HF token. Remaining ~93 files (11475−11382) are NaN-only pairs where MMA has zero non-NaN rows after dropping — expected behavior. HF layout: breeds+scenes in flat results/MetricMetricAlignment/; people in results/MetricMetricAlignment/people/ subdirectory (HF 10k-files/dir limit prevents flat upload). |
| MetricSimilarityAlignment (MSA) | 181 | 225 | 🔴 80% | 1×3×3×5mp×5s |
| MetricSimilarityAlignmentMulti (MSAM) | 112 | 90 | ✅ DONE | 1×3×3×5mp×2 reg_algos; >expected means extra exploratory runs on disk |
| InterferenceMatrix (IM) | 45 | 45 | ✅ DONE | 1×3×3×5mp |
| SimilarityMatrix (SM) | 13 | 15 | 🔴 87% | 1×3×5s; weight_overlap scenes/distil only so actual < 15 |
| SignificantRelationshipCategorical (SRC) | 4248 | N/A | — | N/A — count depends on which (me, attr, task, method) combos have sufficient samples |
| SignificantRelationshipCategoricalDirectional (SRCD) | 106 | N/A | — | N/A — run for specific directional pairs; no fixed expected |
| SignificantRelationshipNumerical (SRN) | 387 | N/A | — | N/A — data-dependent dispatch (numerical attrs only) |
| ImplicitAssociationTest (IAT) | 6 | 18 | 🔴 33% | people-only × 3×2lat×3 attr_pairs; actual may be less if embeddings missing for some combos |
| MethodComparisonByMetricEntity (MCME) | 158 | 153 | ✅ DONE | 1×3×51me; extra if exploratory runs on disk |
| EmbeddingUnlearningProfile (EUP) | 543 | 900 | 🔴 60% | 1×3×3×100entities; actual < expected if embedding files missing for some method/entity |
| EmbeddingForgettingEfficiency (EFE) | 9 | 9 | ✅ DONE | 1×3×3 |
| InterferenceVisualSummary (IVS) | 3 | N/A | — | Not in standard pipeline_08 run (requires HF image download) |
| MetricSimilarityAlignmentOne (MSAOne) | 9 | N/A | — | On-demand; run for specific entity+method+metric combinations |
| MinimumCutInterference (MCI) | 23 | N/A | — | On-demand; run for a hand-picked set of 10 combinations |

**Total JSON files across all RT directories:** 14593

## Directory listing

Raw file count per directory (including non-standard directories):

| Directory | JSON files |
|-----------|------------|
| `CountSignificantRelationship` | 3 |
| `EmbeddingForgettingEfficiency` | 9 |
| `EmbeddingUnlearningProfile` | 543 |
| `ImplicitAssociationTest` | 6 |
| `InterferenceMatrix` | 45 |
| `InterferenceVisualSummary` | 3 |
| `MethodComparisonByMetricEntity` | 158 |
| `MetricMetricAlignment` | 8747 |
| `MetricSimilarityAlignment` | 181 |
| `MetricSimilarityAlignmentMulti` | 112 |
| `MetricSimilarityAlignmentOne` | 9 |
| `MinimumCutInterference` | 23 |
| `SignificantRelationshipCategorical` | 4248 |
| `SignificantRelationshipCategoricalDirectional` | 106 |
| `SignificantRelationshipNumerical` | 387 |
| `SimilarityMatrix` | 13 |

