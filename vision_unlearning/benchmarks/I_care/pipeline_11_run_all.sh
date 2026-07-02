# Run the full I-CARE pipeline end-to-end.
# All scripts are idempotent: re-running skips already-completed steps.
# Parameters must be consistent with configuration.py.
# Total compute: >1000 GPU-hours; meant as a reference, not a single run.
clear
cd vision-unlearning/vision_unlearning/benchmarks/I_care

# pipeline_01 is a Jupyter notebook (.ipynb) — execute with:
#   jupyter nbconvert --to notebook --execute pipeline_01_get_data.ipynb
jupyter nbconvert --to notebook --execute pipeline_01_get_data.ipynb

python pipeline_02_compute_similarities.py --task breeds --similarity clip
python pipeline_02_compute_similarities.py --task breeds --similarity jacc
python pipeline_02_compute_similarities.py --task breeds --similarity dino
python pipeline_02_compute_similarities.py --task breeds --similarity act
python pipeline_02_compute_similarities.py --task scenes --similarity clip
python pipeline_02_compute_similarities.py --task scenes --similarity jacc
python pipeline_02_compute_similarities.py --task scenes --similarity dino
python pipeline_02_compute_similarities.py --task scenes --similarity act
python pipeline_02_compute_similarities.py --task people --similarity clip
python pipeline_02_compute_similarities.py --task people --similarity jacc
python pipeline_02_compute_similarities.py --task people --similarity dino
python pipeline_02_compute_similarities.py --task people --similarity act

python pipeline_03_unlearn_model.py --task breeds --method uce --num-train-epochs 0
python pipeline_03_unlearn_model.py --task breeds --method munba --num-train-epochs 50
python pipeline_03_unlearn_model.py --task breeds --method distil --num-train-epochs 100
python pipeline_03_unlearn_model.py --task scenes --method uce --num-train-epochs 0
python pipeline_03_unlearn_model.py --task scenes --method munba --num-train-epochs 100
python pipeline_03_unlearn_model.py --task scenes --method distil --num-train-epochs 100
python pipeline_03_unlearn_model.py --task people --method uce --num-train-epochs 0
python pipeline_03_unlearn_model.py --task people --method munba --num-train-epochs 200
python pipeline_03_unlearn_model.py --task people --method distil --num-train-epochs 400

# Baseline dataset (method-agnostic; run once per task before the per-method runs)
python pipeline_04_generate_dataset.py --task breeds --baseline
python pipeline_04_generate_dataset.py --task scenes --baseline
python pipeline_04_generate_dataset.py --task people --baseline

# Per-method generated datasets (unlearned model → LoRA-ON + LoRA-OFF images)
python pipeline_04_generate_dataset.py --task breeds --method uce --num-train-epochs 0
python pipeline_04_generate_dataset.py --task breeds --method munba --num-train-epochs 50
python pipeline_04_generate_dataset.py --task breeds --method distil --num-train-epochs 100
python pipeline_04_generate_dataset.py --task scenes --method uce --num-train-epochs 0
python pipeline_04_generate_dataset.py --task scenes --method munba --num-train-epochs 100
python pipeline_04_generate_dataset.py --task scenes --method distil --num-train-epochs 100
python pipeline_04_generate_dataset.py --task people --method uce --num-train-epochs 0
python pipeline_04_generate_dataset.py --task people --method munba --num-train-epochs 200
python pipeline_04_generate_dataset.py --task people --method distil --num-train-epochs 400

python pipeline_05_compute_embeddings.py --embedding dino --task breeds --method uce --num-train-epochs 0
python pipeline_05_compute_embeddings.py --embedding dino --task breeds --method munba --num-train-epochs 50
python pipeline_05_compute_embeddings.py --embedding dino --task breeds --method distil --num-train-epochs 100
python pipeline_05_compute_embeddings.py --embedding dino --task scenes --method uce --num-train-epochs 0
python pipeline_05_compute_embeddings.py --embedding dino --task scenes --method munba --num-train-epochs 100
python pipeline_05_compute_embeddings.py --embedding dino --task scenes --method distil --num-train-epochs 100
python pipeline_05_compute_embeddings.py --embedding dino --task people --method uce --num-train-epochs 0
python pipeline_05_compute_embeddings.py --embedding dino --task people --method munba --num-train-epochs 200
python pipeline_05_compute_embeddings.py --embedding dino --task people --method distil --num-train-epochs 400

python pipeline_06_compute_interference_per_pair.py --task breeds --method uce --num-train-epochs 0
python pipeline_06_compute_interference_per_pair.py --task breeds --method munba --num-train-epochs 50
python pipeline_06_compute_interference_per_pair.py --task breeds --method distil --num-train-epochs 100
python pipeline_06_compute_interference_per_pair.py --task scenes --method uce --num-train-epochs 0
python pipeline_06_compute_interference_per_pair.py --task scenes --method munba --num-train-epochs 100
python pipeline_06_compute_interference_per_pair.py --task scenes --method distil --num-train-epochs 100
python pipeline_06_compute_interference_per_pair.py --task people --method uce --num-train-epochs 0
python pipeline_06_compute_interference_per_pair.py --task people --method munba --num-train-epochs 200
python pipeline_06_compute_interference_per_pair.py --task people --method distil --num-train-epochs 400


python pipeline_07_compute_interference_per_entity.py --task breeds --methods uce munba distil --epochs 0 50 100
python pipeline_07_compute_interference_per_entity.py --task scenes --methods uce munba distil --epochs 0 100 100
python pipeline_07_compute_interference_per_entity.py --task people --methods uce munba distil --epochs 0 200 400

python pipeline_08_run_all_rts.py --rts MetricMetricAlignment MetricSimilarityAlignment InterferenceMatrix SimilarityMatrix SignificantRelationship CountSignificantRelationship MethodComparisonByMetricEntity ImplicitAssociationTest 

# Upload everything computed above; pass --no-upload for a check-only completion report.
python pipeline_09_synchronize_huggingface.py --stages metadata models generated-datasets embeddings act-fingerprints interference-per-pair interference-per-entity

python pipeline_10_generate_paper_results.py