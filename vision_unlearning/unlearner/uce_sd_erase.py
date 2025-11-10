from __future__ import annotations
import os
import time
import copy
import logging
from pathlib import Path
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any, cast
from PIL import Image


import torch  # noqa: F401
import torch.nn as nn
from pydantic import Field
from safetensors.torch import save_file, load_file
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from huggingface_hub.repocard_data import EvalResult
from huggingface_hub import upload_folder

from vision_unlearning.unlearner.base import Unlearner, logger
from vision_unlearning.evaluator import EvaluatorTextToImage
from vision_unlearning.metrics import MetricImageTextSimilarity
from vision_unlearning.utils.model_management import save_model_card


class ConceptType(str, Enum):
    """Enum representing the type of concept to unlearn."""
    Object = "object"
    Art = "art"


class UCE(Unlearner):
    """
    Unified Concept Editing for unlearning in Stable Diffusion models.
    Adapted from:
        GitHub: https://github.com/rohitgandikota/unified-concept-editing
        Arxiv: https://arxiv.org/pdf/2308.14761.pdf
        Gandikota, R., Orgad, H., Belinkov, Y., Materzyńska, J., & Bau, D. (2024).
        Unified concept editing in diffusion models. In Proceedings of the IEEE/CVF
        Winter Conference on Applications of Computer Vision (pp. 5111-5120).
    This unlearner do not use LoRA, and do not perform any fine-tuning (instead, it performs a closed-form weight update).
    """

    # Specific to this unlearner
    pretrained_model_name_or_path: str = Field(
        default="CompVis/stable-diffusion-v1-4",
        description="Path to pretrained model or model identifier from huggingface.co/models."
    )
    erase_scale: float = Field(0.5, description="Must be positive. Higher = more aggressive erasure?? TODO explain")
    preserve_scale: float = Field(1.0, description="Must be non-negative. Higher = more careful preservation?? TODO explain")
    lamb: float = Field(0.5, description="Must be between 0 and 1. Higher = ?? TODO explain")
    save_entire_model: bool = Field(True, description="Whether to save the entire model or just the updated parts.")

    # Dataset related
    edit_concepts: Optional[str] = None
    guide_concepts: Optional[str] = None
    preserve_concepts: Optional[str] = None
    concept_type: ConceptType = Field(
        default=ConceptType.Object,
        description="Type of concept to unlearn."
    )
    expand_prompts: bool = True
    final_eval_prompts_forget: str | List[str] = Field([], description="Prompts for final evaluation on the forget dataset (ModelHub identifier or directly the prompts).")
    final_eval_prompts_retain: str | List[str] = Field([], description="Prompts for final evaluation on the retain dataset (ModelHub identifier or directly the prompts).")

    # Other stuff (some for compatibility with UnlearnerLora)
    output_dir: str = Field(
        default="../uce_models",
        description="Output directory for model predictions and checkpoints."
    )
    device: str = "cuda:0"
    compute_runtimes: bool = Field(True, description="Whether to compute the runtimes of the training, for evaluation purposes.")
    hub_model_id: Optional[str] = Field(None, description="Repository name to sync with `output_dir`. None for not push")

    def __init__(self, **data: Any):
        """Custom initializer for UCE with informative logging."""
        super().__init__(**data)

        logger.info("\n[INFO] Initializing Unified Concept Eraser (UCE)...")
        logger.info(f" - Base model:        {self.pretrained_model_name_or_path}")
        logger.info(f" - Device:            {self.device}")
        logger.info(f" - Erase scale:       {self.erase_scale}")
        logger.info(f" - Preserve scale:    {self.preserve_scale}")
        logger.info(f" - Regularization λ:  {self.lamb}")
        logger.info(f" - Edit concepts:     {self.edit_concepts}")
        logger.info(f" - Guide concepts:    {self.guide_concepts}")
        logger.info(f" - Preserve concepts: {self.preserve_concepts}")
        logger.info(f" - Concept type:      {self.concept_type}")
        logger.info(f" - Output directory:  {self.output_dir}")
        logger.info(f" - Expand prompts:    {self.expand_prompts}\n")
        logger.info(f" - Compute runtimes:  {self.compute_runtimes}\n")
        logger.info(f" - Hub Model Id:      {self.hub_model_id}\n")

    def _collect_text_embeddings(
        self,
        pipe: Any,
        concepts: list[str],
        device: str,
        torch_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        """Return dict {concept: last_token_embedding}."""
        uce_embeds: dict[str, torch.Tensor] = {}

        for e in concepts:
            if e in uce_embeds:
                continue
            t_emb = pipe.encode_prompt(
                prompt=e,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            last_token_idx = (
                pipe.tokenizer(
                    e,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )["attention_mask"]
            ).sum() - 2

            uce_embeds[e] = t_emb[0][:, last_token_idx, :]
        return uce_embeds

    def _collect_guide_outputs(
        self,
        concepts: list[str],
        embeds: dict[str, torch.Tensor],
        modules: list[torch.nn.Module]
    ) -> dict[str, list[torch.Tensor]]:
        """Collect cross-attention outputs for guide/preserve concepts."""
        outputs: dict[str, list[torch.Tensor]] = {}

        for g in concepts:
            if g in outputs:
                continue
            t_emb = embeds[g]
            for module in modules:
                outputs[g] = outputs.get(g, []) + [module(t_emb)]
        return outputs

    def _update_weights(
        self,
        original_modules: list[torch.nn.Module],
        erase_embeds: dict[str, torch.Tensor],
        guide_outputs: dict[str, list[torch.Tensor]],
        edit_concepts: list[str],
        guide_concepts: list[str],
        preserve_concepts: list[str],
        erase_scale: float,
        preserve_scale: float,
        lamb: float,
        device: str,
        torch_dtype: torch.dtype
    ) -> list[torch.nn.Module]:
        """Apply the UCE weight update to each module and return new modules."""
        uce_modules = copy.deepcopy(original_modules)

        for module_idx, module in enumerate(original_modules):
            if isinstance(module, nn.Module):
                w_old: torch.Tensor = cast(torch.Tensor, module.weight)
            else:
                w_old = cast(torch.Tensor, module)  # fallback if somehow not a module

            # Compute mat1 safely
            mat1: torch.Tensor = lamb * w_old

            # Compute mat2 safely
            mat2: torch.Tensor = lamb * torch.eye(
                w_old.shape[1], device=w_old.device, dtype=w_old.dtype
            )

            # Erase concepts
            for erase_concept, guide_concept in zip(edit_concepts, guide_concepts):
                c_i = erase_embeds[erase_concept].T
                v_i_star = guide_outputs[guide_concept][module_idx].T
                mat1 += erase_scale * (v_i_star @ c_i.T)
                mat2 += erase_scale * (c_i @ c_i.T)

            # Preserve concepts
            for preserve_concept in preserve_concepts:
                c_i = erase_embeds[preserve_concept].T
                v_i_star = guide_outputs[preserve_concept][module_idx].T
                mat1 += preserve_scale * (v_i_star @ c_i.T)
                mat2 += preserve_scale * (c_i @ c_i.T)

            # uce_modules[module_idx].weight = torch.nn.Parameter(
            #     mat1 @ torch.inverse(mat2.float()).to(torch_dtype)
            # )

            eps = 1e-6
            mat2_float = mat2.float() + eps * torch.eye(mat2.shape[0], device=mat2.device)
            uce_modules[module_idx].weight = torch.nn.Parameter(
                (mat1 @ torch.inverse(mat2_float)).to(torch_dtype)
            )

        return uce_modules

    def _save_uce_weights(
        self,
        uce_modules: list[torch.nn.Module],
        uce_module_names: list[str],
    ) -> None:
        """Save updated module weights to a safetensors file."""
        uce_state_dict: dict[str, torch.Tensor] = {}
        for name, parameter in zip(uce_module_names, uce_modules):
            weight_tensor: torch.Tensor = cast(torch.Tensor, parameter.weight)
            uce_state_dict[name + ".weight"] = weight_tensor

        # Just modified weights
        save_file(uce_state_dict, os.path.join(self.output_dir, "uce_sd_weights.safetensors"))

        # Entire model
        if self.save_entire_model:
            pipe = self.get_pipeline_from_modified_weights()
            pipe.save_pretrained(self.output_dir)  # type: ignore

    def train(self) -> List[EvalResult]:
        """Main UCE training and concept erasure logic."""

        # ==== Sanity checks ====
        t0 = time.time()
        assert self.pretrained_model_name_or_path, "Pretrained model path must not be empty."
        assert isinstance(self.erase_scale, (int, float)) and self.erase_scale > 0, "Erase scale must be positive."
        assert isinstance(self.preserve_scale, (int, float)) and self.preserve_scale >= 0, "Preserve scale must be non-negative."
        assert 0.0 <= self.lamb <= 1.0, "Lambda must be between 0 and 1."
        assert self.device in ["cpu", "cuda", "cuda:0", "cuda:1"], f"Invalid device specified: {self.device}"
        assert isinstance(self.concept_type, ConceptType), "concept_type must be of type ConceptType Enum."

        if isinstance(self.final_eval_prompts_retain, str):
            raise NotImplementedError("final_eval_prompts_retain should be a list of prompts, not a string.")
        if isinstance(self.final_eval_prompts_forget, str):
            raise NotImplementedError("final_eval_prompts_forget should be a list of prompts, not a string.")

        if "cuda" in self.device:
            assert torch.cuda.is_available(), "CUDA device specified but not available!"

        if not self.save_entire_model:
            raise NotImplementedError("UCE currently only supports saving the entire model. Set save_entire_model to True.")

        torch_dtype: torch.dtype = torch.float32
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        if self.pretrained_model_name_or_path != "CompVis/stable-diffusion-v1-4":
            logging.warning("UCE was not tested with this base model; results may differ.")

        t1 = time.time()

        # ==== Concept parsing ====
        assert self.edit_concepts, "At least one edit concept must be provided."
        edit_list: list[str] = [c.strip() for c in self.edit_concepts.split(';') if c.strip()]

        assert len(edit_list) > 0, "Edit concepts list cannot be empty after parsing."

        guide_list: list[str] = []
        if self.guide_concepts:
            guide_list = [c.strip() for c in self.guide_concepts.split(';') if c.strip()]
        elif self.concept_type == ConceptType.Art:
            guide_list = ["art"] * len(edit_list)
        else:
            # default guide for objects: use a neutral object class or same as edit concept
            guide_list = [e for e in edit_list]

        if len(guide_list) == 1:
            guide_list *= len(edit_list)

        if len(guide_list) != len(edit_list):
            raise ValueError(
                "Mismatch between edit and guide concepts. Ensure they are separated by ';' and have equal counts."
            )

        preserve_list: list[str] = []
        if self.preserve_concepts:
            preserve_list = [c.strip() for c in self.preserve_concepts.split(';') if c.strip()]

        # ==== Prompt expansion ====
        if self.expand_prompts:
            edit_copy = copy.deepcopy(edit_list)
            guide_copy = copy.deepcopy(guide_list)

            for concept, guide_concept in zip(edit_copy, guide_copy):
                if self.concept_type == ConceptType.Art:
                    edit_list.extend([
                        f"painting by {concept}", f"art by {concept}",
                        f"artwork by {concept}", f"picture by {concept}",
                        f"style of {concept}"
                    ])
                    guide_list.extend([
                        f"painting by {guide_concept}", f"art by {guide_concept}",
                        f"artwork by {guide_concept}", f"picture by {guide_concept}",
                        f"style of {guide_concept}"
                    ])
                else:
                    edit_list.extend([
                        f"image of {concept}", f"photo of {concept}",
                        f"portrait of {concept}", f"picture of {concept}",
                        f"painting of {concept}", f"picture of {concept} doing something"
                    ])
                    guide_list.extend([
                        f"image of {guide_concept}", f"photo of {guide_concept}",
                        f"portrait of {guide_concept}", f"picture of {guide_concept}",
                        f"painting of {guide_concept}", f"picture of {concept} doing something"
                    ])

        t2 = time.time()
        logger.info(f"\nErasing: {edit_list}\nGuiding: {guide_list}\nPreserving: {preserve_list} with erase_scale: {self.erase_scale}, preserve_scale: {self.preserve_scale} and regularization lambda: {self.lamb}\n")  # noqa

        # ==== Weight update ====
        pipe = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        ).to(self.device)

        torch.set_grad_enabled(False)

        # Find relevant modules
        uce_modules: list[torch.nn.Module] = []
        uce_module_names: list[str] = []

        for name, module in pipe.unet.named_modules():
            if "attn2" in name and (name.endswith("to_v") or name.endswith("to_k")):
                uce_modules.append(module)
                uce_module_names.append(name)

        assert len(uce_modules) > 0, "No attention modules found for UCE to operate on."
        original_modules = copy.deepcopy(uce_modules)

        # Collect embeddings
        all_concepts = edit_list + guide_list + preserve_list
        erase_embeds = self._collect_text_embeddings(pipe, all_concepts, self.device, torch_dtype)
        assert all(c in erase_embeds for c in all_concepts), "Some concepts failed to produce embeddings."

        # Collect guide outputs
        guide_outputs = self._collect_guide_outputs(guide_list + preserve_list, erase_embeds, original_modules)

        # Apply weight updates
        updated_modules = self._update_weights(
            original_modules, erase_embeds, guide_outputs,
            edit_list, guide_list, preserve_list,
            self.erase_scale, self.preserve_scale, self.lamb, self.device, torch_dtype
        )

        # ==== Post training ====
        t3 = time.time()
        self._save_uce_weights(updated_modules, uce_module_names)
        eval_results, eval_images = self.evaluate()
        t4 = time.time()

        metric_common_attributes = {
            "task_type": "text-to-image",
            "dataset_type": f"forget-and-retain-together",
            "dataset_name": f"{edit_list} (forget) and {preserve_list} (retain) sets",
        }

        if self.compute_runtimes:
            eval_results.append(EvalResult(
                metric_type='runtime',
                metric_name=f'Runtime init seconds (~↓)',
                metric_value=t1 - t0,
                **metric_common_attributes,  # type: ignore
            ))
            eval_results.append(EvalResult(
                metric_type='runtime',
                metric_name=f'Runtime data loading seconds (~↓)',
                metric_value=t2 - t1,
                **metric_common_attributes,  # type: ignore
            ))
            eval_results.append(EvalResult(
                metric_type='runtime',
                metric_name=f'Runtime training seconds (↓)',
                metric_value=t3 - t2,
                **metric_common_attributes,  # type: ignore
            ))
            eval_results.append(EvalResult(
                metric_type='runtime',
                metric_name=f'Runtime eval seconds (~↓)',
                metric_value=t4 - t3,
                **metric_common_attributes,  # type: ignore
            ))

        save_model_card(
            str(self.hub_model_id),
            images=eval_images,
            base_model=self.pretrained_model_name_or_path,
            dataset_forget_name=self.edit_concepts,
            dataset_retain_name=str(self.preserve_concepts),
            repo_folder=self.output_dir,
            eval_results=eval_results,
            tags=[
                "stable-diffusion",
                "stable-diffusion-diffusers",
                "text-to-image",
                "diffusers",
                "diffusers-training",
            ],
            hyperparameters={k: v for k, v in self.model_dump().items() if (isinstance(v, (str, float, int, type(None))) and not isinstance(v, (Enum)))},
        )  # type: ignore[arg-type]

        if self.hub_model_id is not None:
            upload_folder(
                repo_id=self.hub_model_id,
                folder_path=self.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        return eval_results

    def get_pipeline_from_modified_weights(self) -> DiffusionPipeline:
        pipe = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)

        logger.debug("Base model is loaded.\n")

        uce_state_dict = load_file(os.path.join(self.output_dir, "uce_sd_weights.safetensors"))
        logger.debug(f"Loaded {len(uce_state_dict)} UCE weight tensors")

        # Applying the modified weights
        with torch.no_grad():
            for name, param in pipe.unet.named_parameters():
                if name in uce_state_dict:
                    logger.debug(f"Updating: {name}")
                    param.copy_(uce_state_dict[name])

        return pipe

    def evaluate(self) -> Tuple[List[EvalResult], Dict[str, Image.Image]]:
        assert type(self.final_eval_prompts_forget) == list  # noqa
        assert type(self.final_eval_prompts_retain) == list  # noqa
        if self.save_entire_model:
            pipeline_original = AutoPipelineForText2Image.from_pretrained(self.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker=None).to(self.device)
            pipeline_unlearned = AutoPipelineForText2Image.from_pretrained(self.output_dir, torch_dtype=torch.float16, safety_checker=None).to(self.device)
        else:
            # Maybe have a static method that loads the base model and applies the uce weights?
            # Should be easy for the user to do the same
            raise NotImplementedError("UCE evaluation currently only supports saving the entire model. Set save_entire_model to True.")

        evaluator = EvaluatorTextToImage(
            pipeline_original=pipeline_original,
            pipeline_unlearned=pipeline_unlearned,
            pipeline_learned=None,
            prompts_forget=self.final_eval_prompts_forget,
            prompts_retain=self.final_eval_prompts_retain,
            metric_clip=MetricImageTextSimilarity(metrics=['clip']),
            compute_runtimes=self.compute_runtimes
        )

        eval_result, eval_images = evaluator.evaluate()

        return eval_result, eval_images
