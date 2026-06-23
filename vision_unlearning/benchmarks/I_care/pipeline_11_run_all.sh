# Single script to run everything in the I_care study
# This can be very impractical, since the actual execution take more a 1000 GPU hours
# However, since all scripts are idenpotent, this is a reasonable way of checking if everything was actually completely executed
# Most of these invocations parameters have to match what is declared in `configuration.py`

# TODO: the following invocations dont quite work yet (some arguments are actually hardcoded in the scripts, among other minor problems)
clear
cd vision-unlearning/vision_unlearning/benchmarks/I_care

python pipeline_01_get_data.py

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


python pipeline_07_compute_interference_per_entity.py --task breeds --methods "uce,munba,distil" --num-train-epochs-list "0,50,100"
python pipeline_07_compute_interference_per_entity.py --task scenes --methods "uce,munba,distil" --num-train-epochs-list "0,100,100"
python pipeline_07_compute_interference_per_entity.py --task people --methods "uce,munba,distil" --num-train-epochs-list "0,200,400"

python pipeline_08_run_all_rts.py

python pipeline_09_synchronize_huggingface.py

python pipeline_10_generate_paper_results.py