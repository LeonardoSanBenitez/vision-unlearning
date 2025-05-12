import os
import time
from typing import List, Dict, Tuple
from pydantic import BaseModel, ConfigDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from huggingface_hub.repocard_data import EvalResult

from vision_unlearning.metrics import MetricImageTextSimilarity, MetricPaintingStyle, FrechetInceptionDistance
from vision_unlearning.utils.logger import get_logger
if is_wandb_available():
    from vision_unlearning.integrations.wandb import wandb_log_image
from vision_unlearning.integrations.tensorboard import tensorboard_log_image

from vision_unlearning.utils.images_handling import verify_images_in_path, ValidImageExtensions


logger = get_logger('evaluation')


class EvaluatorTextToImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path_to_original_model: Optional[str] = None                # path to the original model
    pipeline_original: Optional[StableDiffusionPipeline] = None # original pipeline
    gen_path_original_forget: Optional[str] = None              # path to generated images for forget set
    gen_path_original_retain: Optional[str] = None              # path to generated images for retain set

    path_to_unlearned_model: Optional[str] = None                   # path to the unlearned model
    pipeline_unlearned: Optional[StableDiffusionPipeline] = None    # pipeline to be evaluated
    gen_path_unlearned_retain: Optional[str] = None                 # path to generated images for retain set
    gen_path_unlearned_forget: Optional[str] = None                 # path to generated images for forget set
    
    prompts_forget: List[str] = None  # prompts to evaluate forget set -> CLIP
    prompts_retain: List[str] = None   # prompts to evaluate retain set -> CLIP
    real_path_forget: str = None       # path to real images of forget set
    real_path_retain: str = None       # path to real images of retain set

    path_to_save_outputs: Optional[str] = None  # path to save the generated images
    # metric_clip: MetricImageTextSimilarity
    # metric_fid: FrechetInceptionDistance
    compute_runtimes: bool = True

    valid_extensions = ValidImageExtensions._member_names_

    def model_post_init(self, __context: dict = None) -> None:

        # verify if necessary parameters are set
        # needs real images for FID
        assert self.real_path_retain is not None and self.real_path_forget is not None,\
            "Could not find real images data!\r\nPlease define a path to a folder with retain and another with forget images."
        
        if self.real_path_retain:
            assert verify_images_in_path(self.real_path_retain), \
                f"No valid images found in the folder '{self.real_path_retain}'."
        
        if self.real_path_forget:
            assert verify_images_in_path(self.real_path_forget), \
                f"No valid images found in the folder '{self.real_path_forget}'."
        
        # verify it has the necessary prompts or/and generated images to compute CLIP and FID
        # for CLIP it needs the generated images and the prompts
        assert self.gen_path_forget is not None and self.prompts_forget is not None,\
            "Could not find generated images data!\r\nPlease define a path to a folder or a torch.Tensor with the images."
        
        if self.gen_path_forget:
            assert verify_images_in_nested_path(self.gen_path_forget), \
                f"No valid images found in the folder '{self.gen_path_forget}'."
        
        assert self.gen_path_retain is not None and self.prompts_retain is not None,\
            "Could not find generated images data!\r\nPlease define a path to a folder or a torch.Tensor with the images."    
        
        if self.gen_path_retain:
            assert self.verify_images_in_nested_path(self.gen_path_retain), \
                f"No valid images found in the folder '{self.gen_path_retain}'."
        
        assert self.path_to_original_model is not None or self.pipeline_original is not None, \
            "Could not find the original model!\r\nPlease define a path to the original model or a StableDiffusionPipeline."

        assert self.path_to_unlearned_model is not None or self.pipeline_unlearned is not None, \
            "Could not find the unlearned model!\r\nPlease define a path to the unlearned model or a StableDiffusionPipeline."
        
        if self.path_to_save_outputs is not None:
            assert os.path.exists(self.path_to_save_outputs),\
                f"The path '{self.path_to_save_outputs}' does not exist."
            assert os.path.isdir(path_to_save_outputs),\
                f"The path '{self.path_to_save_outputs}' is not a folder."
        
        pass

    def verify_images_in_nested_path(self, folder_path: str) -> bool:
        """
        Verifies if the given path contains image files in nested folders.
        :param folder_path: Path to the folder to check.
        :return: True if images are found, False otherwise.
        """
        original_folder = os.path.join(folder_path, 'original')
        unlearned_folder = os.path.join(folder_path, 'unlearned')
        folders = [original_folder, unlearned_folder]

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The path '{folder_path}' does not exist.")
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"The path '{folder_path}' is not a directory.")
        
        count = 0
        for folder in folders:
            if not os.path.exists(folder):  # check if the original folder exists
                raise FileNotFoundError(f"The path '{folder}' does not exist.")
            if not os.path.isdir(folder):  # check if the original folder is a directory
                raise NotADirectoryError(f"The path '{folder}' is not a directory.")

            
            for file in os.listdir(folder):
                if os.path.splitext(file)[1][1:].upper() in valid_extensions:
                    count += 1
                    break  # stop checking after finding the first valid image
        
        if count >= len(folders):  # check if all folders have at least one valid image
            return True
        return False


    def load_images_from_folder(self, folder_path: str, subpath: List[str] = ['original','unlearned']) -> List[Image.Image]:
        """
        Loads all images from a folder and returns them as a list of PIL Images.
        :param folder_path: Path to the folder containing images.
        :return: A list of PIL Images.
        """
        images = {}

        folders = [os.path.join(folder_path, f) for f in subpath]

        for folder in folders:
            sub
            images[os.path.split(folder)[1]] = []  # Initialize the list for each folder

            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                if os.path.splitext(file_name)[1][1:].upper() in valid_extensions:
                    try:
                        image = Image.open(file_path).convert('RGB')  # Ensure 3-channel RGB
                        os.path.split(folder)[1]  # Get the folder name
                        images[os.path.split(folder)[1]].append(image)
                    except Exception as e:
                        logger.error(f"Error loading image {file_name}: {e}")

        return image_list

    def evaluate(self) -> Tuple[List[EvalResult], Dict[str, Image.Image]]:

        final_metrics: dict = {}  # JSON structure to store metrics for each image
        gen_images: dict = {}

        # create file structure for saving outputs
        if self.path_to_save_outputs is not None:
            os.mkdir(os.path.join(self.path_to_save_outputs, 'original'), exist_ok=True)
            os.mkdir(os.path.join(self.path_to_save_outputs, 'original', 'forget'), exist_ok=True)
            os.mkdir(os.path.join(self.path_to_save_outputs, 'original', 'retain'), exist_ok=True)

            os.mkdir(os.path.join(self.path_to_save_outputs, 'unlearned'), exist_ok=True)
            os.mkdir(os.path.join(self.path_to_save_outputs, 'unlearned', 'forget'), exist_ok=True)
            os.mkdir(os.path.join(self.path_to_save_outputs, 'unlearned', 'retain'), exist_ok=True)

        # generate images for forget and retain sets and compute CLIP scores
        for scope, prompts in {'forget': self.prompts_forget, 'retain': self.prompts_retain}.items():

            # verify if number of prompts match number of generated images
            # if scope == 'forget':
            #     if self.gen_path_forget is not None:
            #         loaded_imgs = self.load_images_from_folder(self.gen_path_forget)
            #         assert len(prompts) == len(loaded_imgs), \
            #             f"Number of prompts ({len(prompts)}) does not match number of generated images ({len(loaded_imgs)})."
                    
            # else:
            #     if self.gen_path_retain is not None:
            #         assert len(prompts) == len(os.listdir(self.gen_path_retain)), \
            #             f"Number of prompts ({len(prompts)}) does not match number of generated images ({len(os.listdir(self.gen_path_retain))})."
            
            clip_scores_original: List[float] = []
            clip_scores_unlearned: List[float] = []
            scores_difference_original_unlearned: List[float] = []
            latencies_original: List[float] = []
            latencies_unlearned: List[float] = []

            image_metrics = {}

            for i, prompt in enumerate(prompts):
                t0 = time.time()
                image_original = self.pipeline_original(prompt).images[0]  # type: ignore
                latencies_original.append(time.time() - t0)
                t0 = time.time()
                image_unlearned = self.pipeline_unlearned(prompt).images[0]  # type: ignore
                latencies_unlearned.append(time.time() - t0)

                score_original = self.metric_clip.score(image_original, prompt)['clip']
                score_unlearned = self.metric_clip.score(image_unlearned, prompt)['clip']
                scores_original.append(score_original)
                scores_unlearned.append(score_unlearned)
                scores_difference_original_unlearned.append(score_original - score_unlearned)

                # TODO: implement a visualization function based on data saved
                # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                # axes[0].imshow(image_original)
                # axes[0].set_title(f"Original\nClip Score={score_original:.2f}")
                # axes[0].axis("off")
                # axes[1].imshow(image_learned)
                # axes[1].set_title(f"Learned\nClip Score={score_learned:.2f}")
                # axes[1].axis("off")
                # axes[2].imshow(image_unlearned)
                # axes[2].set_title(f"Unlearned\nClip Score={score_unlearned:.2f}")
                # axes[2].axis("off")
                # fig.suptitle(prompt, fontsize=16)
                # fig.canvas.draw()
                # images[prompt] = Image.fromarray(np.uint8(np.array(fig.canvas.buffer_rgba())))  # type: ignore
                # plt.show()

                gen_images[f"{scope}_original"].append(image_original)
                gen_images[f"{scope}_unlearned"].append(image_unlearned)

                # save images to the path
                if self.path_to_save_outputs is not None:
                    # TODO: check if the images need to be denormalized and uint8
                    image_original.save(os.path.join(self.path_to_save_outputs, 'original', scope, f"{prompt}_{i}.png"))
                    image_unlearned.save(os.path.join(self.path_to_save_outputs, 'unlearned', scope, f"{prompt}_{i}.png"))
            
                # Save metrics for the current image
                image_metrics[f"{scope}_{i}"] = {
                    "prompt": f"{prompt}_{i}",
                    "CLIP_original": score_original,
                    "CLIP_unlearned": score_unlearned,
                    "CLIP_difference": score_original - score_unlearned,
                    "latency_original": latencies_original[-1],
                    "latency_unlearned": latencies_unlearned[-1]
                }
            
            final_metrics[f"{scope}_set"] = {
                "individual_metrics": image_metrics,
                "num_of_images": len(prompts),
                "average_CLIP_original": np.mean(scores_original),
                "std_CLIP_original": np.std(scores_original),
                "average_CLIP_unlearned": np.mean(scores_unlearned),
                "std_CLIP_unlearned": np.std(scores_unlearned),
                "average_CLIP_difference": np.mean(scores_difference_original_unlearned),
                "average_latency_original": np.mean(latencies_original),
                "std_latency_original": np.std(latencies_original),
                "average_latency_unlearned": np.mean(latencies_unlearned),
                "std_latency_unlearned": np.std(latencies_unlearned)
            }
            if self.path_to_save_outputs is not None:
                with open(os.path.join(self.path_to_save_outputs, 'metrics.json'), 'w') as f:
                    json.dump(final_metrics, f, indent=4)

        return final_metrics, gen_images