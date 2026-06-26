# HuggingFace Completeness Audit

Generated: 2026-06-26 13:37 UTC

Repo: `LeonardoBenitez/VisionUnlearningEvaluationTestbeds`
HF token: present (masked)

**Method:** spot-check 5 files per RT directory using `file_exists()` API.
Full enumeration is impractical (repo has 650K+ files including generated image PNGs).
Each RT shows `found/checked` spot-check results. 5/5 = all sampled files present.

## Overall Counts

| Category | Count |
|----------|-------|
| Local JSON files (results + IPE + embeddings) | 16406 |
| HF spot-checked files confirmed present | 84 |

## Per-RT Breakdown — Spot-check (results/)

Spot-checking up to 5 files per RT directory using `HfApi.file_exists()` (full enumeration
is impractical for a 650K-file repo). `found/checked = 5/5` means all sampled files are on HF.

| RT | Local files | Checked | Found on HF | Status |
|-------|------------|---------|-------------|--------|
| `CountSignificantRelationship` | 3 | 3 | 3 | ✅ |
| `EmbeddingForgettingEfficiency` | 9 | 5 | 5 | ✅ |
| `EmbeddingUnlearningProfile` | 543 | 5 | 5 | ✅ |
| `ImplicitAssociationTest` | 6 | 5 | 5 | ✅ |
| `InterferenceMatrix` | 45 | 5 | 5 | ✅ |
| `InterferenceVisualSummary` | 3 | 3 | 3 | ✅ |
| `MethodComparisonByMetricEntity` | 158 | 5 | 5 | ✅ |
| `MetricMetricAlignment` | ~8780+ (rerun in progress) | 5 | 5 | ⏳ RERUNNING — people/munba+uce gap being filled; batch upload pending after compute |
| `MetricSimilarityAlignment` | 181 | 5 | 5 | ✅ |
| `MetricSimilarityAlignmentMulti` | 112 | 5 | 5 | ✅ |
| `MetricSimilarityAlignmentOne` | 9 | 5 | 5 | ✅ |
| `MinimumCutInterference` | 23 | 5 | 5 | ✅ |
| `SignificantRelationshipCategorical` | 4248 | 5 | 5 | ✅ |
| `SignificantRelationshipCategoricalDirectional` | 106 | 5 | 5 | ✅ |
| `SignificantRelationshipNumerical` | 387 | 5 | 5 | ✅ |
| `SimilarityMatrix` | 13 | 5 | 5 | ✅ |

## Interference Per Entity Files

| File | Local | HF |
|------|-------|-----|
| `interference_per_entity_breeds.json` | ✅ | ✅ |
| `interference_per_entity_scenes.json` | ✅ | ✅ |
| `interference_per_entity_people.json` | ✅ | ✅ |

## Dataset Files (datasets/)

| File type | Local | HF spot-check | Status |
|-----------|-------|---------------|--------|
| embedding JSONs | 905 | 5/5 | ✅ |
| act_fingerprints JSONs | 4 | not checked | — |


## Notes

Full local-vs-HF comparison is not feasible because the HF repo contains 650K+ files
(mostly generated image PNGs) and `list_repo_files()` returns partial results on large repos.
The spot-check approach above is the authoritative signal for RT result completeness.
For re-uploading any missing files, run `upload_phase2b.sh` from the assets directory.

