import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import tqdm
import argparse
# from PIL import Image
from constants import theme_available, class_available

all_unlearned = theme_available + class_available

# =========================
# CONFIG
# =========================
base_model_id = "runwayml/stable-diffusion-v1-5"
# model_path = "/home/molna/old_diffusers/examples/text_to_image/unlearn_canvas/main_checkpoint"
# lora_path = "/home/molna/old_diffusers/examples/text_to_image/unlearn_canvas/ckpts_bz_to_bluebloom/checkpoint-1500"
# output_dir = "/home/molna/old_diffusers/examples/text_to_image/unlearn_canvas/generated_bz_to_bluebloom"

use_subfolders = False

device = "cuda"
dtype = torch.float16


def generate_images(args, theme):
    global device, dtype, base_model_id

    guidance_scale = 7.5
    width = height = 512

    # =========================
    # PIPELINE
    # =========================
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    ).to(device)

    pipe.unet = UNet2DConditionModel.from_pretrained(
        args.pipeline_path,
        subfolder="unet",  # or "unet_ema"
        torch_dtype=dtype,
    ).to(device)

    lora_path = os.path.join(args.ckpts_base_path, theme, "pytorch_lora_weights.safetensors")
    pipe.load_lora_weights(lora_path)
    pipe.safety_checker = None

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Xformers enabled.")
    except Exception as e:
        print(f"Could not enable Xformers: {e}. Performance will be slower.")

    pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=True)

    # create output directory for theme
    output_dir = os.path.join(args.output_dir, theme)
    os.makedirs(output_dir, exist_ok=True)


    for style in tqdm.tqdm(theme_available, desc="Generating styles"):
        style_dir = output_dir
        if use_subfolders:
            style_dir = os.path.join(output_dir, style)
            os.makedirs(style_dir, exist_ok=True)

        for obj in class_available:
            
            style_name = "Photo" if style == "Seed_Images" else style
            prompt = f"An image of {obj} in {style_name.replace('_', ' ')} style."

            prompts = [prompt] * len(args.seed)
            generators = [
                torch.Generator(device=device).manual_seed(s)
                for s in args.seed
            ]

            images = pipe(
                prompt=prompts,
                width=width,
                height=height,
                num_inference_steps=args.inf_steps,
                guidance_scale=guidance_scale,
                generator=generators,
            ).images

            for img, seed in zip(images, args.seed):
                filename = f"{style}_{obj}_seed{seed}.jpg"
                save_path = os.path.join(style_dir, filename)
                img.save(save_path)

            print(f"Saved {style} / {obj}")

    print("✅ Finished generating all images.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate_images',
        description='Generate images from unlearned model with FADE.')
    # will run for all theme_available + class_available
    parser.add_argument('--ckpts_base_path', type=str, default="assets/models/", help="Path to the generated distill lora weights should be where the base path where all models are.")
    parser.add_argument('--pipeline_path', help='path to pipeline', type=str, default="assets/data/sd/")
    parser.add_argument('--output_dir', help='folder where to save images', type=str, default="assets/gen_img_samples/")
    parser.add_argument("--seed", type=int, default=[42], nargs="+", required=False, help="Random seed for generation.")
    parser.add_argument('--inf_steps', help='number of inference steps', type=int, required=False, default=30)
    parser.add_argument('--forget_concept', help='Concepts to forget, must be in classes available or themes available.', type=str, nargs="+", required=False, default=all_unlearned)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print("Arguments:", args)

    for theme in args.forget_concept:
        if theme not in all_unlearned:
            print(f"Error: The concept '{theme}' is not in the available themes or classes list.")
            continue
        if theme == "Seed_Images":
            print(f"Error: The concept 'Seed_Images' is excluded from generation. Skipping.")
            continue

        # check if lora weights exist for this concept
        lora_path = os.path.join(args.ckpts_base_path, theme, "pytorch_lora_weights.safetensors")
        if os.path.exists(lora_path):
            print(f"Generating images for concept '{theme}'...")
            generate_images(args, theme)
        else:
            print(f"Error: LoRA weights not found for concept '{theme}'. Skipping.")