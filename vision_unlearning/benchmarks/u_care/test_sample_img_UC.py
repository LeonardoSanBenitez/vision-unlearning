import sys
import argparse

sys.path.append('..')
sys.path.append('../../vision-unlearning')

from generate_unlearned_models import generate_images
from vision_unlearning.unlearner import UnlearnerLoraDistillation

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='test_sample_img_UC',
        description='Generate image answer set for unlearned models with FADE for UnlearnCanvaEval')
    parser.add_argument('--model_base_path', type=str, default="assets/data/sd", required=False, help="Name or path to trained model that should be modified.")
    parser.add_argument('--batch_size', type=int, default=20, required=False, help="Batch size for running batch image generation.")
    parser.add_argument('--save_base_path', type=str, default="assets/UC_test", required=False, help="Folder to save unlearned models.")
    
    args = parser.parse_args()

    generate_images(
        save_base_path=args.save_base_path,
        original_model_path=args.model_base_path,
        batch_size=args.batch_size,
    )

    print("Test was a success!")