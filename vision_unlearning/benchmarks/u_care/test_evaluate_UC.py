import sys
import argparse

sys.path.append('..')
sys.path.append('../../vision-unlearning')

from generate_unlearned_models import summarize_metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='test_evaluation_UC',
        description='Generate accuracy evaluations for answer set of unlearned models with SPARE for UnlearnCanvaEval')
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the folder with ckpt of the UnlearnCanvas classifiers.")
    parser.add_argument('--save_base_path', type=str, required=True, help="Folder where the unlearned models are.")
    parser.add_argument('--uc_dataset_path', type=str, default="assets/data/UC_dataset", required=False, help="Root path to the dataset, where the json files with retain and forget info can be found.")
    parser.add_argument('--batch_size', type=int, default=16, required=False, help="Batch size for running fid evaluation.")
    
    args = parser.parse_args()

    summarize_metrics(
        save_base_path=args.save_base_path,
        ckpt_path=args.ckpt_path,
        uc_dataset_path=args.uc_dataset_path,
        batch_size=args.batch_size,
    )

    print("Test was a success!")