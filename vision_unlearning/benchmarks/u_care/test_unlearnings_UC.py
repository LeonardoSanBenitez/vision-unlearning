import sys
import argparse

sys.path.append('..')
sys.path.append('../../vision-unlearning')

from generate_unlearned_models import run_unlearnings
from vision_unlearning.unlearner import UnlearnerLoraDistillation

hyperparameters = {
    "num_train_epochs": 2,
    "dataloader_num_workers": 2,
    "resolution": 512,
    "num_validation_images": 1,
    "mixed_precision": "no", # specified by model trained on styles for unlearn canvas
    "learning_rate": 1e-4,
    "max_grad_norm": 5.0,
    "lr_warmup_steps": 0,
    "validation_epochs": 1,
    "checkpointing_steps": 10000,
    "lr_scheduler_type": "constant",
    "logging_steps": 20,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "random_flip": True,
    "lora_r": 4,
    "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
    "lora_alpha": 8,
    "lora_dropout": 0.2,
    "seed": 42,
    "dataset_forget_name": "forget",
    "dataset_retain_name": "retain",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='test_unlearnings_UC',
        description='Generate unlearned models with FADE for UnlearnCanvaEval')
    parser.add_argument('--model_base_path', type=str, default="assets/data/sd", required=False, help="Name or path to trained model that should be modified.")
    parser.add_argument('--uc_dataset_path', type=str, default="assets/data/UC_dataset", required=False, help="Root path to the dataset, where the json files with retain and forget info can be found.")
    parser.add_argument('--save_base_path', type=str, default="assets/UC_test", required=False, help="Folder to save unlearned models.")
    
    args = parser.parse_args()

    run_unlearnings(
        UnlearnerLoraDistillation,
        hyperparameters,
        model_base_path=args.model_base_path,
        uc_dataset_path=args.uc_dataset_path,
        save_base_path=args.save_base_path)

    print("Test was a success!")