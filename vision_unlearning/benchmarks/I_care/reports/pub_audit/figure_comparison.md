# Figure Comparison Audit

Generated: 2026-06-26 13:37 UTC  
Updated: 2026-06-26 15:05 UTC — pipeline_10 bug fixes applied (commit b066b08)

Reference: `paper/images/` (81 images)
Generated: `reports/paper_outputs/` (pipeline_10_generate_paper_results.py)

Identity determined by SHA-256 hash comparison.

## Summary

| Status | Count | Meaning |
|--------|-------|---------|
| ✅ IDENTICAL | 0 | Hash matches paper reference |
| ⚠️ DIFFERS | 30 | Generated but hash does not match paper (data + format changes since paper was committed) |
| ❌ MISSING | 0 | Expected from pipeline_10 but not found |
| 🕐 DEFERRED | 0 | All previously deferred figures now generated |
| 📄 STATIC | 51 | Not produced by pipeline_10 (diagrams, screenshots, etc.) |

## Pipeline_10 Bug Fixes Applied (2026-06-26, commit b066b08)

7 figure generation bugs were identified and fixed. All DIFFERS below are expected (data
reconciliation changed SR counts; EUP format differs from paper; etc.) — NOT errors in generation:

1. `iat_hpi.png` — wrong attribute `gender` → fixed to `occupation_simplified`
2. `sig_breeds_group.png` — was bar chart → now SRC boxplot via `ResultTemplateSignificantRelationshipCategorical`
3. `sig_by_me.png` — was bar chart → now Me × method heatmap
4. `sig_by_attribute.png` — missing numerical attrs → added `_load_srn_significant_counts()` for grooming_frequency_value, hpi, birthyear
5. MSAOne rank plots — were calling MSAOne scatter RT → now correctly call `ResultTemplateInterferenceBySimilarityRank`
6. `visual_summary_schnauzer_uce.png` — now generates 4 per-seed images (seed42/43/44/45) separately
7. `latent_dino_winona.png` — method fixed to `distil` (was `munba`); embedding file repaired; EUP generates correctly

**Systematic format difference for latent_dino_* figures:** The paper shows a single-panel PCA scatter colored by per-pair Mp (clip_diff), while the current EUP RT generates a 2-panel figure (PCA + displacement histogram) with aggregated Me coloring. This is an RT format evolution, not a data error. The generated figures are informative but stylistically different from paper.

## Auto-Generated Figures (pipeline_10 scope)

### ⚠️ MethodComparisonByMetricEntity.png

**DIFFERS** — MCME 3-panel (People/Scenes/Breeds)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/MethodComparisonByMetricEntity.png) | ![](../paper_outputs/MethodComparisonByMetricEntity.png) |

### ⚠️ MetricMetricAlignment_clip_rmse.png

**DIFFERS** — MMA: clip_diff vs rmse, UCE, people

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/MetricMetricAlignment_clip_rmse.png) | ![](../paper_outputs/MetricMetricAlignment_clip_rmse.png) |

### ⚠️ SignificantRelationshipCategoricalDirectional_occupation_people_fade.png

**DIFFERS** — SRC Directional: occupation, people, SPARE (individual panel)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/SignificantRelationshipCategoricalDirectional_occupation_people_fade.png) | ![](../paper_outputs/SignificantRelationshipCategoricalDirectional_occupation_people_fade.png) |

### ⚠️ SignificantRelationshipNumerical_birthyear_ssim.png

**DIFFERS** — SRN: birthyear vs ssim, people, SPARE

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/SignificantRelationshipNumerical_birthyear_ssim.png) | ![](../paper_outputs/SignificantRelationshipNumerical_birthyear_ssim.png) |

### ⚠️ SignificantRelationshipNumerical_hpi_rmse.png

**DIFFERS** — SRN: hpi vs rmse, people, SPARE

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/SignificantRelationshipNumerical_hpi_rmse.png) | ![](../paper_outputs/SignificantRelationshipNumerical_hpi_rmse.png) |

### ⚠️ SimilarityMatrix_people_clip.png

**DIFFERS** — SM: people, clip similarity (cosmetic diff: full 100×100 vs paper subset)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/SimilarityMatrix_people_clip.png) | ![](../paper_outputs/SimilarityMatrix_people_clip.png) |

### ⚠️ dinodiff_jacc_scenes_distil.png

**DIFFERS** — MSAOne: DINOv2-diff vs Jaccard, scenes, SPARE

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/dinodiff_jacc_scenes_distil.png) | ![](../paper_outputs/dinodiff_jacc_scenes_distil.png) |

### ⚠️ equalization.png

**DIFFERS** — MMA equalization scatter (people, all methods)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/equalization.png) | ![](../paper_outputs/equalization.png) |

### ⚠️ iat_gender.png

**DIFFERS** — IAT: gender vs occupation_simplified, people

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/iat_gender.png) | ![](../paper_outputs/iat_gender.png) |

### ⚠️ iat_hpi.png

**DIFFERS** — IAT: occupation_simplified vs hpi_bin, people (fixed from wrong `gender` attribute)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/iat_hpi.png) | ![](../paper_outputs/iat_hpi.png) |

### ⚠️ iat_uce.png

**DIFFERS** — IAT: occupation_simplified vs hpi_bin, people, UCE

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/iat_uce.png) | ![](../paper_outputs/iat_uce.png) |

### ⚠️ latent_dino_bush.png

**DIFFERS** — EUP: George W Bush, SPARE

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/latent_dino_bush.png) | ![](../paper_outputs/latent_dino_bush.png) |

### ⚠️ latent_dino_serena.png

**DIFFERS** — EUP: Serena Williams, SPARE

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/latent_dino_serena.png) | ![](../paper_outputs/latent_dino_serena.png) |

### ⚠️ msa_full_groupby_method.png

**DIFFERS** — MSA full grid grouped by method

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/msa_full_groupby_method.png) | ![](../paper_outputs/msa_full_groupby_method.png) |

### ⚠️ msa_full_heatmap_sim_mp_abs.png

**DIFFERS** — MSA full grid abs-correlation heatmap

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/msa_full_heatmap_sim_mp_abs.png) | ![](../paper_outputs/msa_full_heatmap_sim_mp_abs.png) |

### ⚠️ msaone_giant_schnauzer_clip_diff_dino.png

**DIFFERS** — MSAOne scatter: Giant Schnauzer, UCE, clip_diff, dino sim

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/msaone_giant_schnauzer_clip_diff_dino.png) | ![](../paper_outputs/msaone_giant_schnauzer_clip_diff_dino.png) |

### ⚠️ msaone_rank_giant_schnauzer_uce_clip_diff_dino.png

**DIFFERS** — MSAOne rank plot: Giant Schnauzer, UCE, clip_diff, dino

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/msaone_rank_giant_schnauzer_uce_clip_diff_dino.png) | ![](../paper_outputs/msaone_rank_giant_schnauzer_uce_clip_diff_dino.png) |

### ⚠️ msaone_rank_ice_skating_distil_clip_diff_act.png

**DIFFERS** — MSAOne rank plot: Ice Skating, SPARE, clip_diff, act

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/msaone_rank_ice_skating_distil_clip_diff_act.png) | ![](../paper_outputs/msaone_rank_ice_skating_distil_clip_diff_act.png) |

### ⚠️ paretto.png

**DIFFERS** — MMA Pareto scatter (scenes, all methods)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/paretto.png) | ![](../paper_outputs/paretto.png) |

### ⚠️ sig_breeds_group.png

**DIFFERS** — SRC boxplot: breeds, group attribute (fixed from wrong bar chart; data also changed since paper — see SR discrepancy)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/sig_breeds_group.png) | ![](../paper_outputs/sig_breeds_group.png) |

### ⚠️ sig_by_attribute.png

**DIFFERS** — CSR: sig count by attribute (NOTE: differs from paper — see SR discrepancy)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/sig_by_attribute.png) | ![](../paper_outputs/sig_by_attribute.png) |

### ⚠️ sig_by_me.png

**DIFFERS** — Me × method heatmap (fixed from wrong bar chart; data also changed since paper — see SR discrepancy)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/sig_by_me.png) | ![](../paper_outputs/sig_by_me.png) |

### ⚠️ sig_by_method.png

**DIFFERS** — CSR: sig count by method (NOTE: differs from paper — denominator and values changed after Phase 0)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/sig_by_method.png) | ![](../paper_outputs/sig_by_method.png) |

### ⚠️ sig_dir_occupation.png

**DIFFERS** — SRC Directional: occupation, people, 3-panel composite

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/sig_dir_occupation.png) | ![](../paper_outputs/sig_dir_occupation.png) |

### ⚠️ sig_dir_sports.png

**DIFFERS** — SRC Directional: sports, scenes, 2-panel composite

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/sig_dir_sports.png) | ![](../paper_outputs/sig_dir_sports.png) |

### ⚠️ sig_hpi_clip.png

**DIFFERS** — SRN: hpi vs ssim, people, SPARE (historical name says 'clip', metric is ssim)

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/sig_hpi_clip.png) | ![](../paper_outputs/sig_hpi_clip.png) |

### ⚠️ sim_matrix_act.png

**DIFFERS** — SM: act similarity clustermap, occupation-ordered

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/sim_matrix_act.png) | ![](../paper_outputs/sim_matrix_act.png) |

### ⚠️ visual_summary_schnauzer_uce.png

**DIFFERS** — IVS: Giant Schnauzer, UCE (fixed to generate per-seed images: seed42/43/44/45 separately; paper showed composite of all seeds)

| Paper reference | Generated (seed 42) | Generated (seed 43) | Generated (seed 44) | Generated (seed 45) |
|---|---|---|---|---|
| ![](../../../../../../paper/images/visual_summary_schnauzer_uce.png) | ![](../paper_outputs/visual_summary_schnauzer_uce_seed42.png) | ![](../paper_outputs/visual_summary_schnauzer_uce_seed43.png) | ![](../paper_outputs/visual_summary_schnauzer_uce_seed44.png) | ![](../paper_outputs/visual_summary_schnauzer_uce_seed45.png) |

### ⚠️ visual_summary_skating_distil.png

**DIFFERS** — IVS: Ice Skating, SPARE

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/visual_summary_skating_distil.png) | ![](../paper_outputs/visual_summary_skating_distil.png) |

### ⚠️ latent_dino_winona.png

**DIFFERS** — EUP: Winona Ryder, SPARE (fixed from munba → distil; embedding file was repaired)
Format differs: paper shows single-panel per-pair Mp scatter; EUP RT shows 2-panel (PCA + histogram) — systematic difference, see note above.

| Paper reference | Generated output |
|---|---|
| ![](../../../../../../paper/images/latent_dino_winona.png) | ![](../paper_outputs/latent_dino_winona.png) |

## Static Figures (not produced by pipeline_10)

Committed directly to `paper/images/`. No programmatic comparison possible.

| Filename |
|----------|
| `act_person_per_layer.png` |
| `example_distil_people_blair.png` |
| `example_distil_people_gloria.png` |
| `example_distil_people_hewitt.png` |
| `example_munba_breeds_fail.png` |
| `example_uce_scene_velodrome.png` |
| `forgetty_UML_diagrams_components.png` |
| `forgetty_UML_diagrams_sequence.png` |
| `graph.png` |
| `grid_clipdiff_act_top5.png.png` |
| `improvements.png` |
| `interference_example.png` |
| `msa_coloring.png` |
| `msa_grid_clipdiff_act_top5.png` |
| `pipeline_interference.png` |
| `pipeline_testbed.png` |
| `qualitative_breeds_forget_original.png` |
| `qualitative_breeds_forget_unlearned.png` |
| `qualitative_breeds_interfered_original.png` |
| `qualitative_breeds_interfered_unlearned.png` |
| `qualitative_breeds_retain_original.png` |
| `qualitative_breeds_retain_unlearned.png` |
| `qualitative_breeds_retained_original.png` |
| `qualitative_breeds_retained_unlearned.png` |
| `qualitative_people_forget_original.png` |
| `qualitative_people_forget_unlearned.png` |
| `qualitative_people_interfered_original.png` |
| `qualitative_people_interfered_unlearned.png` |
| `qualitative_people_retain_original.png` |
| `qualitative_people_retain_unlearned.png` |
| `qualitative_people_retained_original.png` |
| `qualitative_people_retained_unlearned.png` |
| `qualitative_scenes_forget_original.png` |
| `qualitative_scenes_forget_unlearned.png` |
| `qualitative_scenes_interfered_original.png` |
| `qualitative_scenes_interfered_unlearned.png` |
| `qualitative_scenes_retain_original.png` |
| `qualitative_scenes_retain_unlearned.png` |
| `qualitative_scenes_retained_original.png` |
| `qualitative_scenes_retained_unlearned.png` |
| `results_temp_1.png` |
| `results_temp_4.png` |
| `results_temp_4_old.png` |
| `screen_form.png` |
| `screen_rt_1.png` |
| `screen_rt_2.png` |
| `screen_rt_3.png` |
| `screenshot_list.png` |
| `varied_all.png` |
| `varied_images.png` |
| `varied_one.png` |

