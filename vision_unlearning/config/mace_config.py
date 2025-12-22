# from dataclasses import dataclass
# from typing import List, Optional, Tuple


# @dataclass
# class MACEConfig:
#     # -------------------------
#     # REQUIRED FIELDS (MUST BE FIRST)
#     # -------------------------
#     multi_concept: List[List[Tuple[str, str]]]
#     mapping_concept: List[str]

#     # -------------------------
#     # DEVICE
#     # -------------------------
#     device: str = "cuda"

#     # -------------------------
#     # Primary Settings
#     # -------------------------
#     use_pooler: bool = True
#     train_batch_size: int = 1
#     learning_rate: float = 1e-4
#     max_train_steps: int = 50
#     train_preserve_scale: float = 1e-4
#     fuse_preserve_scale: float = 1e-4
#     augment: bool = True
#     lamb: float = 0.0
#     rank: int = 1
#     lora: bool = True
#     train_separate: bool = True
#     importance_sampling: bool = True
#     max_memory: int = 1000
#     aug_length: int = 30
#     prompt_len: int = 30
#     all_words: bool = False
#     generate_data: bool = True #TO BE REMOVED
#     use_gpt: bool = False
#     test_erased_model: bool = False # TO BE REMOVED

#     # -------------------------
#     # Cache / Preservation
#     # -------------------------
#     prior_preservation_cache_path: str = "./cache/cache_coco.pt"
#     domain_preservation_cache_path: str = "./cache/cache_art.pt"
#     preserve_weight: float = 8.0e4

#     # -------------------------
#     # Paths
#     # -------------------------
#     input_data_dir: str = "./data/100art"
#     output_dir: str = "./saved_model/CFR_with_multi_LoRAs"
#     final_save_path: str = "./saved_model/LoRA_fusion_model"

#     # -------------------------
#     # Grounded-SAM
#     # -------------------------
#     use_gsam_mask: bool = True
#     use_sam_hq: bool = True
#     grounded_config: Optional[str] = None
#     grounded_checkpoint: Optional[str] = None
#     sam_hq_checkpoint: Optional[str] = None
#     sam_checkpoint: Optional[str] = None

#     # -------------------------
#     # Diffusion / Model
#     # -------------------------
#     pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4"
#     with_prior_preservation: bool = False
#     preserve_prompt: str = "a person"
#     preserve_data_dir: str = "data/a_person" 
#     prior_loss_weight: float = 1.0
#     with_uncond_loss: bool = False
#     negative_guidance: float = 1.0
#     uncond_loss_weight: float = 1.0
#     num_class_images: int = 200
#     seed: int = 2024
#     resolution: int = 512
#     revision: Optional[str] = None
#     tokenizer_name: Optional[str] = None
#     instance_prompt: Optional[str] = None
#     concept_keyword: Optional[str] = None
#     no_real_image: bool = False
#     center_crop: bool = False
#     train_text_encoder: bool = False
#     sample_batch_size: int = 4
#     num_train_epochs: int = 1
#     checkpointing_steps: int = 500
#     resume_from_checkpoint: Optional[str] = None
#     gradient_accumulation_steps: int = 1
#     gradient_checkpointing: bool = False
#     scale_lr: bool = False
#     lr_scheduler: str = "constant"
#     lr_warmup_steps: int = 0
#     lr_num_cycles: int = 1
#     lr_power: float = 1.0
#     use_8bit_adam: bool = False
#     dataloader_num_workers: int = 0
#     adam_beta1: float = 0.9
#     adam_beta2: float = 0.999
#     adam_weight_decay: float = 0.01
#     adam_epsilon: float = 1e-8
#     max_grad_norm: float = 1.0
#     push_to_hub: bool = False
#     hub_token: Optional[str] = None
#     hub_model_id: Optional[str] = None
#     logging_dir: str = "logs"
#     allow_tf32: bool = False
#     report_to: str = "tensorboard"
#     mixed_precision: Optional[str] = None
#     prior_generation_precision: Optional[str] = None
#     local_rank: int = -1
#     enable_xformers_memory_efficient_attention: bool = False
#     set_grads_to_none: bool = False
