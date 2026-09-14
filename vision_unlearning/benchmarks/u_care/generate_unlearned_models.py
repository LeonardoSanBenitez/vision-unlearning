import os
import json
import sys
import random
import torch
import gc
import traceback
import tqdm
import timm
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from fid import load_style_generated_images, load_style_ref_images, calculate_fid
sys.path.append('..')
sys.path.append('../../vision-unlearning')

from vision_unlearning.utils.logger import get_logger, setup_loggers
from vision_unlearning.unlearner import UnlearnerSpareSparsePerModule, UnlearnerSpare, UnlearnerSpareSparsePerWeight
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

from constants import class_available, theme_available

def get_all_prompts_from_valid_json(json_file_path, split_name="train"):
    """
    Reads a single, valid JSON file, expects an array of examples under the
    specified split key (e.g., "train"), and extracts all 'text' fields.

    Args:
        json_file_path (str): The full path to the .json file.
        split_name (str): The key in the JSON file that holds the list of examples (default is "train").

    Returns:
        list: A list of strings containing all the extracted prompts.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}")
        return []

    prompts_list = []

    # 1. Read the entire file content
    with open(json_file_path, 'r') as f:
        full_data = json.load(f)  # json.load() reads and parses the entire file

    # 2. Check if the specified split exists in the loaded data
    if split_name not in full_data or not isinstance(full_data[split_name], list):
        print(f"Error: Could not find '{split_name}' list in the JSON data.")
        return []

    # 3. Iterate through the list of dictionaries (the dataset examples)
    for entry in full_data[split_name]:
        if 'text' in entry:
            prompts_list.append(entry['text'])

    return prompts_list

def generate_metadata(images_path: str, forget_concept: str, overwrite_concept: str, metadata_path: str) -> None:
    global theme_available, class_available
    # iterate over the folders to get the paths

    os.makedirs(metadata_path, exist_ok=True)
    
    # create metadata file for forget concept
    metadata_forget = []
    metadata_retain = []

    folders = os.listdir(images_path)

    for theme_name in folders: # first level of UnlearnCanvas data tree, corresponding to styles
        classes = os.listdir(os.path.join(images_path, theme_name))

        for class_name in classes:

            images = os.listdir(os.path.join(images_path, theme_name, class_name))

            if forget_concept == class_name or forget_concept == theme_name:
                # forget set

                if forget_concept == class_name:
                    # replace object concept
                    if theme_name == "Seed_Images":
                        prompt_overwrite = f"An image of {overwrite_concept} in Photo style."
                    else:
                        prompt_overwrite = f"An image of {overwrite_concept} in {theme_name.replace('_', ' ')} style."

                else:                    
                    # replace style concept
                    if overwrite_concept == "Seed_Images":
                        prompt_overwrite = f"An image of {class_name} in Photo style."
                    else:
                        prompt_overwrite = f"An image of {class_name} in {overwrite_concept.replace('_', ' ')} style."
                
                if theme_name == "Seed_Images":
                    prompt_forget = f"An image of {class_name} in Photo style."
                else: 
                    prompt_forget = f"An image of {class_name} in {theme_name.replace('_', ' ')} style."

                for image in images:
                    metadata_forget.append({
                        "file_name": os.path.join(images_path, theme_name, class_name, image),
                        "text": prompt_forget,
                        "overwrite": prompt_overwrite
                    })

            else:
                # retain set
                if theme_name == "Seed_Images":
                    prompt_retain = f"An image of {class_name} in Photo style."    
                else:
                    prompt_retain = f"An image of {class_name} in {theme_name.replace('_', ' ')} style."

                for image in images:
                    metadata_retain.append({
                        "file_name": os.path.join(images_path, theme_name, class_name, image),
                        "text": prompt_retain,
                        "overwrite": prompt_retain
                    })
    
    metadata_data = {
        "forget": metadata_forget,
        "retain": metadata_retain
    }

    metadata_filename = f"metadata-{forget_concept}.json" # Change extension to .json
    with open(os.path.join(metadata_path, metadata_filename), 'w') as f:
        # Use json.dump to write the entire structure (list under "train")
        json.dump(metadata_data, f, indent=4)


def run_unlearnings(
    unlearner_class: UnlearnerSpareSparsePerModule|UnlearnerSpare|UnlearnerSpareSparsePerWeight,
    hyperparameters: dict[str, float|bool|str],
    model_base_path: str,
    uc_dataset_path: str,
    save_base_path: str) -> None:

    global class_available, theme_available

    num_to_select = 2 # just the number of validation images generated when unlearning the model

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    os.makedirs(save_base_path, exist_ok=True)
    
    folders = [f for f in os.listdir(save_base_path) if os.path.isdir(os.path.join(save_base_path, f))]
    gen_list = class_available + theme_available
    gen_list.remove("Seed_Images")

    models_list = gen_list.copy()

    for forget_model in gen_list:

        if forget_model in folders:
            # check if the safetensors file was generated
            files = os.listdir(os.path.join(save_base_path, forget_model))
            if "pytorch_lora_weights.safetensors" in files:
                # remove class or style to generation list
                models_list.remove(forget_model)
    
    logger = get_logger('main')
    setup_loggers(modules_info=['vision_unlearning.'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device != 'cuda':
        logger.info("Unable to run unlearnings in cpu!")
        raise SystemError # TODO: get better error
    
    # gen_list should only contain the list of models not yet generated, or unlearned
    for concept in models_list:
        # select random overwrite class or style
        if concept in class_available:
            possible_concepts = class_available.copy()
            possible_concepts.remove(concept)
            validation_prompt = f"An image of {concept} in Crayon style."
        else:
            possible_concepts = theme_available.copy()
            possible_concepts.remove("Seed_Images")
            possible_concepts.remove(concept)
            validation_prompt = f"An image of Dogs in {concept} style."

        overwrite_concept = random.sample(possible_concepts, 1)[0]

        # create paths for metadata and model saving
        model_output_path = os.path.join(save_base_path, concept)
        os.makedirs(model_output_path, exist_ok=True)
        
        # generate metadata for unlearning
        # generate_metadata(uc_dataset_path, concept, overwrite_concept, model_output_path)
        dataset_metadata = os.path.join(model_output_path, f"metadata-{concept}.json")

        # get some prompts for validating model
        prompts_forget = get_all_prompts_from_valid_json(dataset_metadata, split_name="forget")
        prompts_retain = get_all_prompts_from_valid_json(dataset_metadata, split_name="retain")

        if len(prompts_forget) < num_to_select:
            print(f"Error: The list only contains {len(prompts_forget)} prompts, which is less than the requested {num_to_select}.")
            example_prompts_forget = prompts_forget # Return all prompts if the list is too small
        else:
            example_prompts_forget = random.sample(prompts_forget, num_to_select)

        if len(prompts_retain) < num_to_select:
            print(f"Error: The list only contains {len(prompts_retain)} prompts, which is less than the requested {num_to_select}.")
            example_prompts_retain = prompts_retain # Return all prompts if the list is too small
        else:
            example_prompts_retain = random.sample(prompts_retain, num_to_select)

        logger.info(f"Overwriting {concept} by {overwrite_concept}")

        # select batch according to available memory
        free_memory, total_memory = torch.cuda.mem_get_info()  # in GB
        logger.info(f"Free Memory: {free_memory / 1e9:.2f} GB")
        logger.info(f"Total Memory: {total_memory / 1e9:.2f} GB")

        # gradient_accumulation_steps can be bigger
        if free_memory > 20e9: # a100
            logger.info('Choosing hyperparams for free_memory>20e9')
            hyperparameters.update({
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
            })
        elif free_memory > 14e9:  # v100
            logger.info('Choosing hyperparams for free_memory>14e9')
            hyperparameters.update({
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 2,
            })
        else:
            logger.error('You should not even be trying...')

        logger.info(hyperparameters)

        unlearner = unlearner_class(
            model_name_or_path=model_base_path,
            output_dir=model_output_path,
            json_metafile=dataset_metadata,
            image_column="file_name",
            caption_column="text",
            overwrite_column="overwrite",
            validation_prompt=validation_prompt,
            final_eval_prompts_forget = example_prompts_forget,
            final_eval_prompts_retain = example_prompts_retain,
            gradient_weighting_method = GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
            device=device,
            hub_model_id = None,
            **hyperparameters,
        )

        try:
            eval_results = unlearner.train()
            logger.info(f"Eval results: {eval_results}")
            
            del unlearner
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"Finished training for forgetting: {concept}")
        
        except Exception as e:
            logger.info(f"Training for forgetting {concept} was skipped or failed.")
            print(f"{str(e)}\n\n{traceback.print_exc()}")
        
        break

def create_job_queue(output_dir: str, seeds: list[int]) -> list[dict[str, int|str]]:
    global class_available, theme_available

    all_jobs: dict[str, int|str] = []

    print("Preparing job queue...")
    for style in theme_available:

        for obj in class_available:
            # Determine prompt
            if style == "Seed_Images":
                prompt = f"An image of {obj} in Photo style"
            else:
                prompt = f"An image of {obj} in {style.replace('_', ' ')} style"

            for seed in seeds:
                # Store all necessary info to generate and save later
                filename = f"{style}_{obj}_seed{seed}.jpg"
                save_path = os.path.join(output_dir, filename)
                
                job = {
                    "prompt": prompt,
                    "seed": seed,
                    "save_path": save_path,
                    "style": style, # stored for logging/debugging
                    "obj": obj
                }
                all_jobs.append(job)

    total_jobs = len(all_jobs)
    print(f"Total images to generate: {total_jobs}")
    return all_jobs, total_jobs


def generate_images_model(
    base_original_path: str,
    unlearned_base_path: str,
    concept: str,
    device: str,
    batch_size: int,
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    seed: list[int] = [188, 288, 588, 688, 888],
    guidance_scale: float = 7.5,
    inference_steps: int = 25,
    width: int = 512,
    height: int = 512,
    dtype: str = torch.float16
) -> None:
    global class_available, theme_available

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        use_karras_sigmas=True
    )

    pipe.unet = UNet2DConditionModel.from_pretrained(
        base_original_path,
        subfolder="unet",  # or "unet_ema"
        torch_dtype=dtype,
    ).to(device)

    lora_path = os.path.join(unlearned_base_path, "pytorch_lora_weights.safetensors")
    pipe.load_lora_weights(lora_path)
    pipe.safety_checker = None

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Xformers enabled.")
    except Exception as e:
        print(f"Could not enable Xformers: {e}. Performance will be slower.")

    pipe.set_progress_bar_config(disable=True)

    # create output directory for saving images
    output_dir = os.path.join(unlearned_base_path, "img_answer_set")
    os.makedirs(output_dir, exist_ok=True)

    # perform batch generation of images
    all_jobs, total_jobs = create_job_queue(output_dir, seed)

    last_style = None

    for i in tqdm.tqdm(range(0, total_jobs, batch_size), desc="Processing Batches"):
        # Slice the list to get the current batch
        batch_jobs = all_jobs[i : i + batch_size]
        
        # Extract prompts and seeds for this specific batch
        batch_prompts = [job["prompt"] for job in batch_jobs]
        
        # Create a list of generators, one for each image in the batch
        batch_generators = [
            torch.Generator(device=device).manual_seed(job["seed"]) 
            for job in batch_jobs
        ]

        # Run the pipeline once for the whole batch
        # The pipeline handles differing prompts and seeds automatically
        images = pipe(
            prompt=batch_prompts,
            width=width,
            height=height,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=batch_generators,
        ).images

        # Save images to their correct pre-calculated paths
        for img, job in zip(images, batch_jobs):
            img.save(job["save_path"])

        if last_style != batch_jobs[0]["style"]:
            last_style = batch_jobs[0]["style"]
            print(f"✅ Finished generating all images for style: {last_style}")

    print("✅ Finished generating all images.")


def generate_images(
    save_base_path: str,
    original_model_path: str,
    original_model_id: str = "runwayml/stable-diffusion-v1-5",
    batch_size: int = 20,
) -> None:
    global class_available, theme_available

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    os.makedirs(save_base_path, exist_ok=True)
    
    folders = [f for f in os.listdir(save_base_path) if os.path.isdir(os.path.join(save_base_path, f))]
    gen_list = class_available + theme_available
    gen_list.remove("Seed_Images")

    logger = get_logger('main')
    setup_loggers(modules_info=['vision_unlearning.'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models_list = gen_list.copy()

    for forget_model in gen_list:

        if forget_model not in folders:
            logger.info(f"Could not find unlearned model for forgetting {forget_model}")
            models_list.remove(forget_model)
        else:
            # check if the safetensors file was generated
            files = os.listdir(os.path.join(save_base_path, forget_model))
            if "pytorch_lora_weights.safetensors" not in files:
                # remove class or style from list if cannot find the safetensors file
                models_list.remove(forget_model)  

    if device != 'cuda':
        print("Unable to run unlearnings in cpu!")
        raise SystemError # TODO: get better error
    
    # gen_list should only contain the list of models already unlearned for generating images answer set
    for concept in models_list:
        generate_images_model(
            base_original_path = original_model_path,
            unlearned_base_path = os.path.join(save_base_path, concept),
            concept = concept,
            device = device,
            batch_size = batch_size,
            base_model_id = original_model_id,
            seed = [188, 288, 588, 688, 888]
        )
        break

def summarize_metrics(
    save_base_path: str,
    ckpt_path: str,
    uc_dataset_path: str,
    batch_size: int = 16,
    multiprocessing: bool = False
) -> None:
    global class_available, theme_available

    def filter_style_obj_img(img_list: list[str], style: str, obj: str) -> list[str]:
        prefix = f"{style}_{obj}_"
        imgs_filtered = [img for img in img_list if img.startswith(prefix)]
        return imgs_filtered
    
    tasks = ["class", "style"]
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    folders = [f for f in os.listdir(save_base_path) if os.path.isdir(os.path.join(save_base_path, f))]
    gen_list = class_available + theme_available
    gen_list.remove("Seed_Images")

    logger = get_logger('main')
    setup_loggers(modules_info=['vision_unlearning.'])

    models_list = gen_list.copy()

    for forget_model in gen_list:

        if forget_model not in folders:
            logger.info(f"Could not find model for unlearning {forget_model}")
            models_list.remove(forget_model)
        else:
            # verify if there is a answer set of images
            if os.path.exists(os.path.join(save_base_path, forget_model, 'img_answer_set')):
                # plus, the folder should contain 5100 images
                gen_images = [img_file for img_file in os.listdir(os.path.join(save_base_path, forget_model, 'img_answer_set'))]
                if len(gen_images) != 5100:
                    logger.info(f"Could not find all generated images for model {forget_model}")
                    models_list.remove(forget_model)
            else:
                logger.info(f"Could not find generated image folder for unlearning {forget_model}")
                models_list.remove(forget_model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device != 'cuda':
        print("Unable to run unlearnings in cpu!")
        raise SystemError # TODO: get better error
    
    logger.info(f"\nGenerating evaluations for: {models_list}")
    
    # gen_list should only contain the list of models already unlearned for generating images answer set
    for concept in models_list:
        # get seeds list
        image_files = [img_file for img_file in os.listdir(os.path.join(save_base_path, concept, 'img_answer_set')) if img_file.endswith(".jpg")]
        output_dir = os.path.join(save_base_path, concept)

        for task in tasks:
            output_path = os.path.join(output_dir, f"{concept}_{task}_eval.pth")

            model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(device)
            num_classes = len(theme_available) if task == "style" else len(class_available)
            ckpt = os.path.join(ckpt_path, "style50.pth") if task == "style" else os.path.join(ckpt_path, "style50_cls.pth")

            model.head = torch.nn.Linear(1024, num_classes).to(device)
            # load checkpoint
            model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
            model.eval()

            # Initialize misclassification record in the results dictionary
            results = {}
            results["test_theme"] = concept
            if task == "style":
                results["loss"] = {theme: 0.0 for theme in theme_available}
                results["acc"] = {theme: 0.0 for theme in theme_available}
                results["pred_loss"] = {theme: 0.0 for theme in theme_available}
                results["misclassified"] = {theme: {other_theme: 0 for other_theme in theme_available} for theme in theme_available}
            else:
                results["loss"] = {class_: 0.0 for class_ in class_available}
                results["acc"] = {class_: 0.0 for class_ in class_available}
                results["pred_loss"] = {class_: 0.0 for class_ in class_available}
                results["misclassified"] = {class_: {other_class: 0 for other_class in class_available} for class_ in class_available}

            
            image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

            if task == "style":
                for idx, test_theme in tqdm.tqdm(enumerate(theme_available)):
                    theme_label = idx
                    for object_class in class_available:
                        image_filtered = filter_style_obj_img(image_files, test_theme, object_class)
                        for img_name in image_filtered:
                            img_path = os.path.join(save_base_path, concept, 'img_answer_set', img_name)
                            image = Image.open(img_path)
                            target_image = image_transform(image).unsqueeze(0).to(device)
                            with torch.no_grad():
                                res = model(target_image)
                                label = torch.tensor([theme_label]).to(device)
                                loss = torch.nn.functional.cross_entropy(res, label)
                                # softmax the prediction
                                res_softmax = torch.nn.functional.softmax(res, dim=1)
                                pred_loss = res_softmax[0][theme_label]
                                pred_label = torch.argmax(res)
                                pred_success = (torch.argmax(res) == theme_label).sum()

                            results["loss"][test_theme] += loss
                            results["pred_loss"][test_theme] += pred_loss
                            results["acc"][test_theme] += (pred_success * 1.0 / (len(class_available) * len(image_filtered)))

                            misclassified_as = theme_available[pred_label.item()]
                            results["misclassified"][test_theme][misclassified_as] += 1

                    torch.save(results, output_path)

            else:
                for test_theme in tqdm.tqdm(theme_available):
                    for idx, object_class in enumerate(class_available):
                        image_filtered = filter_style_obj_img(image_files, test_theme, object_class)
                        for img_name in image_filtered:
                            theme_label = idx
                            img_path = os.path.join(save_base_path, concept, 'img_answer_set', img_name)
                            image = Image.open(img_path)
                            target_image = image_transform(image).unsqueeze(0).to(device)
                            with torch.no_grad():
                                res = model(target_image)
                                label = torch.tensor([theme_label]).to(device)
                                loss = torch.nn.functional.cross_entropy(res, label)
                                # softmax the prediction
                                res_softmax = torch.nn.functional.softmax(res, dim=1)
                                pred_loss = res_softmax[0][theme_label]
                                pred_success = (torch.argmax(res) == theme_label).sum()
                                pred_label = torch.argmax(res)

                            results["loss"][object_class] += loss
                            results["pred_loss"][object_class] += pred_loss
                            results["acc"][object_class] += (pred_success * 1.0 / (len(theme_available) * len(image_filtered)))
                            misclassified_as = class_available[pred_label.item()]
                            results["misclassified"][object_class][misclassified_as] += 1

                    torch.save(results, output_path)
        
        # compute fid
        images2 = load_style_generated_images(os.path.join(save_base_path, concept, 'img_answer_set'), concept)
        images1 = load_style_ref_images(uc_dataset_path, concept)
        fid_value = calculate_fid(images1, images2, multiprocessing, batch_size)
        print(fid_value)

        torch.save(fid_value, os.path.join(output_dir, f"{concept}_fid_eval.pth"))