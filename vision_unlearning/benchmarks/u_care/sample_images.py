from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import argparse
import os
import sys
sys.path.append("..")

from constants import theme_available, class_available


all_possibilities = theme_available + class_available

def generate_images(args, forgetting_theme):
    output_directory = os.path.join(args.output_dir, forgetting_theme)
    os.makedirs(output_directory, exist_ok=True)

    # get trained lora weights path for defined theme
    if theme_available.__contains__(forgetting_theme):
        ckpt = os.path.join(args.ckpts_base_path, f"forget_style_{forgetting_theme}", f"distill_refactoring_{args.num_train_epochs:03d}")
    else:
        ckpt = os.path.join(args.ckpts_base_path, f"forget_class_{forgetting_theme}", f"distill_refactoring_{args.num_train_epochs:03d}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    pipeline = AutoPipelineForText2Image.from_pretrained(args.pipeline_path, torch_dtype=torch_dtype, safety_checker=None).to(device)

    # load unet lora weights
    pipeline.load_lora_weights(ckpt, weight_name='pytorch_lora_weights.safetensors')  # load attention processors
    pipeline.set_progress_bar_config(disable=True)
    
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        try:
            autocast_ctx = torch.autocast(device.type)  # type: ignore
        except Exception as e:
            print(f'Error: {e}, could not configure autocast_ctx.')
    
    with autocast_ctx:
        if args.seed is not None:
            if not isinstance(args.seed, int):

                # create generators for seeds
                generators = [
                    torch.Generator(device=device).manual_seed(seed)
                    for seed in args.seed
                ]

                for test_theme in theme_available:
                    for object_class in class_available:
                        prompts = [f"A {object_class} image in {test_theme.replace('_', ' ')} style."] * len(args.seed)
                        output_paths = [os.path.join(output_directory, f"{test_theme}_{object_class}_seed{seed}.jpg") for seed in args.seed]

                        images = pipeline(prompts, num_inference_steps=args.inf_steps, generator=generators).images
                        for img, out_path in zip(images, output_paths):
                            if os.path.exists(out_path):
                                print(f"Detected! Skipping {out_path}")
                                continue
                            img.save(out_path)
            else:
                # Set seed
                generator = torch.Generator(device=device).manual_seed(args.seed)
                for test_theme in theme_available:
                    for object_class in class_available:
                        output_path = os.path.join(output_directory, f"{test_theme}_{object_class}_seed{args.seed}.jpg")
                        if os.path.exists(output_path):
                            print(f"Detected! Skipping {output_path}")
                            continue
                        prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
                        image = pipeline(prompt, num_inference_steps=args.inf_steps, generator=generator).images[0]
                        image.save(output_path) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sample_images',
        description='Generate images from unlearned model with SPARE.')
    # will run for all theme_available + class_available
    parser.add_argument('--ckpts_base_path', type=str, default="assets/models/", help="Path to the generated distill lora weights should be where the base path where all models are.")
    parser.add_argument('--num_train_epochs', type=int, default=1 , help="Number of training epochs from the unlearning process.")
    parser.add_argument('--pipeline_path', help='path to pipeline', type=str, default="assets/data/sd/")
    parser.add_argument('--output_dir', help='folder where to save images', type=str, default="assets/gen_img_samples/")
    parser.add_argument("--seed", type=int, default=[42], nargs="+", required=False, help="Random seed for generation.")
    parser.add_argument('--inf_steps', help='number of inference steps', type=int, required=False, default=30)
    parser.add_argument('--model_dtype', help='model dtype', type=str, required=False, default="fp16")
    parser.add_argument('--forget_concept', help='Concepts to forget, must be in classes available or themes available.', type=str, nargs="+", required=False, default=all_possibilities)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    for theme in args.forget_concept:
        if theme not in all_possibilities:
            print(f"Error: The concept '{theme}' is not in the available themes or classes list.")
            continue
        generate_images(args, theme)