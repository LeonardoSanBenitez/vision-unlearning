OUTPUT_MODELS_DIR ?= assets/models_debug
FORGET_CONCEPT ?= Crayon
# FORGET_CONCEPT ?= Statues
REPLACE_CONCEPT ?= Cubism


myloradist_real.py \
						--pretrained_model_name_or_path "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/assets/data/sd" \
						--replacer $(REPLACE_CONCEPT) \
						--forget_concept $(FORGET_CONCEPT) \
						--train_data_dir "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/assets/data/metadata/metadata-$(FORGET_CONCEPT)-retain.json" \
						--forget_data_dir "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/assets/data/metadata/metadata-$(FORGET_CONCEPT)-forget.json" \
						--num_validation_images 1 \
						--output_dir "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/$(OUTPUT_MODELS_DIR)" \
						--seed 42 \
						--resolution 512 \
						--random_flip \
						--train_batch_size $(TRAIN_BATCH_SZ) \
						--gradient_accumulation_steps $(GRAD_ACCUM) \
						--num_train_epochs 1 \
						--max_train_steps 1550 \
						--learning_rate 1e-4 \
						--lr_scheduler "constant" \
						--lr_warmup_steps 0 \
						--dataloader_num_workers 2 \
						--max_grad_norm 1.0 \
						--mixed_precision "no" \
						--checkpointing_steps 10000 \
						--checkpoints_total_limit 2 \
						--rank 4 \
						--lora_alpha 4 \
						--lora_dropout 0.2 \


# generate unlearned models
train_models_style.py \
						--output_dir $(OUTPUT_MODELS_DIR) \
						--themes $(FORGET_CONCEPT)
# train_models_style.py --output_dir assets/models_best --mix_prec no --lr 6e-4 --max_grad_norm 5.0 --val_epochs 2 --scheduler_type constant --lora_r 16 --lora_alpha 4 --lora_dropout 0.2 --themes Abstractionism Artist_Sketch Blossom_Season Bricks Byzantine Cartoon Cold_Warm Color_Fantasy Comic_Etch Crayon Cubism Dadaism Dapple
# train_models_style.py --output_dir assets/models_best --mix_prec no --lr 6e-4 --max_grad_norm 5.0 --val_epochs 2 --scheduler_type constant --lora_r 16 --lora_alpha 4 --lora_dropout 0.2 --themes Defoliation Early_Autumn Expressionism Fauvism French Glowing_Sunset Gorgeous_Love Greenfield Impressionism Ink_Art Joy Liquid_Dreams
# train_models_style.py --output_dir assets/models_best --mix_prec no --lr 6e-4 --max_grad_norm 5.0 --val_epochs 2 --scheduler_type constant --lora_r 16 --lora_alpha 4 --lora_dropout 0.2 --themes Magic_Cube Meta_Physics Meteor_Shower Monet Mosaic Neon_Lines On_Fire Pastel Pencil_Drawing Picasso Pop_Art Red_Blue_Ink Rust Seed_Images
# train_models_style.py --output_dir assets/models_best --mix_prec no --lr 6e-4 --max_grad_norm 5.0 --val_epochs 2 --scheduler_type constant --lora_r 16 --lora_alpha 4 --lora_dropout 0.2 --themes Sketch Sponge_Dabbed Structuralism Superstring Surrealism Ukiyoe Van_Gogh Vibrant_Flow Warm_Love Warm_Smear Watercolor Winter

train_models_object.py --output_dir assets/models_best --mix_prec no --lr 6e-4 --max_grad_norm 5.0 --val_epochs 2 --scheduler_type constant --lora_r 16 --lora_alpha 4 --lora_dropout 0.2 --classes Statues Towers Trees Waterfalls


OUTPUT_GEN_IMGS ?= assets/gen_img_samples_best
CKPTS_BASE_PATH ?= assets/models_debug
PIPELINE_PATH ?= assets/data/sd
OUTPUT_ACCURACY ?= assets/accuracy_results
TASK ?= class
CKPT_CLASSIFIER ?= assets/data/ckpts/style50_cls.pth
sample_images.py \
						--seed 188 288 588 688 888 \
						--forget_concept $(FORGET_CONCEPT)



# img-generation
generate.py \
						--seed 188 288 588 688 888 \
						--ckpts_base_path "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/$(CKPTS_BASE_PATH)" \
						--pipeline_path "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/$(PIPELINE_PATH)" \
						--output_dir "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/$(OUTPUT_GEN_IMGS)" \
						--forget_concept $(FORGET_CONCEPT) \


accuracy.py \
						--seed 188 288 588 688 888 \
						--input_dir "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/$(OUTPUT_GEN_IMGS)" \
						--ckpt "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/$(CKPT_CLASSIFIER)" \
						--task $(TASK) \
						--output_dir "/home/${SSH_USER_CLUSTER}/$(GIT_REPOSITORY_NAME)/UnlearnCanvaEval/$(OUTPUT_ACCURACY)" \
						--forget_concept $(FORGET_CONCEPT)
# accuracy.py --forget_concept Blossom_Season Bricks Byzantine Cartoon Cold_Warm Color_Fantasy Comic_Etch Crayon Cubism Dadaism Dapple Defoliation Early_Autumn Expressionism Fauvism French Glowing_Sunset Gorgeous_Love Pastel Pencil_Drawing --input_dir assets/gen_img_samples --output_dir assets/accuracy_results/ --seed 188 288 588 688 888 --ckpt assets/data/ckpts/style50_cls.pth --task class
# accuracy.py --forget_concept Picasso Pop_Art Red_Blue_Ink Rust Sketch Sponge_Dabbed Structuralism Superstring Surrealism Ukiyoe Van_Gogh Vibrant_Flow Warm_Love Warm_Smear Watercolor Winter --input_dir assets/gen_img_samples --output_dir assets/accuracy_results/ --seed 188 288 588 688 888 --ckpt assets/data/ckpts/style50_cls.pth --task class

# fid.py --p1 assets/data/images --p2 assets/gen_img_samples --output_path assets/accuracy_results --batch_size 32