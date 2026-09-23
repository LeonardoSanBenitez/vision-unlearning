import pytest
import warnings
import torch
from diffusers import UNet2DConditionModel
from vision_unlearning.unlearner.spare import make_lora_elastic, ElasticLoRALinear, SharedR


warnings.filterwarnings("ignore", category=DeprecationWarning, module="peft.*")

@pytest.fixture(scope="module")
def unet():
    return UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")


@pytest.fixture
def r_values():
    return [4, 8, 16]


@pytest.fixture
def target_modules():
    return ["to_q", "to_k", "to_v", "to_out"]


# These tests sometimes hang forever... I suspect this is due to loading the model from CompVis/stable-diffusion-v1-4
'''
def test_module_replacement(unet, r_values, target_modules):
    """
    Test that all target Linear modules (lora_A/lora_B) are replaced by ElasticLoRALinear.
    """
    model = make_lora_elastic(unet, r_values, target_modules, share_rank_within_layer=True)
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and ("lora_A" in name or "lora_B" in name):
            assert isinstance(module, ElasticLoRALinear), f"Module {name} was not replaced."


def test_all_replacements_have_shared_r(unet, r_values, target_modules):
    """
    Check that all ElasticLoRALinear modules in the same layer share the same SharedR instance.
    """
    model = make_lora_elastic(unet, r_values, target_modules, share_rank_within_layer=True)
    shared_r_groups = {}
    for name, module in model.named_modules():
        if isinstance(module, ElasticLoRALinear):
            key = ".".join(name.split(".")[:-1])
            if key not in shared_r_groups:
                shared_r_groups[key] = module.shared_r
            else:
                assert shared_r_groups[key] is module.shared_r, f"Module {name} does not share r correctly"


def test_config_file_created(tmp_path, unet, r_values, target_modules):
    """
    Ensure a config JSON file is created with correct structure when config_save_dir is provided.
    """
    save_dir = tmp_path / "config"
    make_lora_elastic(unet, r_values, target_modules, share_rank_within_layer=True, config_save_dir=save_dir)
    config_path = save_dir / "elastic_adapter_config.json"
    assert config_path.exists(), "Config file not created"
    import json
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, list)
    assert all("target" in entry and "search_space" in entry for entry in config)
'''


def test_state_dict_saves_original_weights():
    """
    Verify that state_dict excludes masked weights (_weight, _bias) and keeps original names.
    """
    r_values = [4, 8, 16]
    layer = ElasticLoRALinear(32, 64, SharedR(r_values), is_lora_A=True)
    sd = layer.state_dict()
    for key in sd.keys():
        assert not key.startswith("_"), f"State dict contains internal key: {key}"


########################################
# What the distillation pair is conditioned on
########################################
def _spare(**overrides):  # type: ignore[no-untyped-def]
    """A SPARE unlearner configured far enough to answer a question about its captions.

    Nothing is loaded and nothing is trained: the two methods under test are pure functions of the
    configuration, which is exactly why they were given names of their own.
    """
    from vision_unlearning.unlearner.spare import UnlearnerSpare
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    settings = {
        'model_name_or_path': 'CompVis/stable-diffusion-v1-4',
        'dataset_forget_name': 'unused/forget',
        'dataset_retain_name': 'unused/retain',
        'num_train_epochs': 1,
        'max_train_steps': 1,
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 1,
        'lr_warmup_steps': 0,
        'resolution': 64,
        'output_dir': 'unused',
        'hub_model_id': None,
        'device': 'cpu',
        'gradient_weighting_method': GradientWeightingMethodSimple(),
        'compute_runtimes': False,
        'compute_gradient_conflict': False,
        'compute_memory': False,
        'overwritting_concept': 'a cat',
        'validation_prompt': None,
        'num_validation_images': 0,
        'final_eval_prompts_forget': [],
        'final_eval_prompts_retain': [],
    }
    settings.update(overrides)
    return UnlearnerSpare(**settings)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "task, class_key",
    [
        ("people", "Mark_Philippoussis"),
        ("breeds", "bouvier des flandres dog"),
        ("scenes", "football_field"),
    ],
)
def test_the_forget_caption_is_the_string_the_split_stores(task, class_key):  # type: ignore[no-untyped-def]
    """The forget side is conditioned on the template applied to the concept, and the training
    split stores that same string for the same entity. These are built by two different modules,
    so nothing but a test keeps them equal -- and if they drift, the distillation pair is fitted at
    a point in text-embedding space that no image was ever captioned with."""
    from vision_unlearning.benchmarks.I_care.split_captions import expected_caption
    from vision_unlearning.datasets.entity_names import canonical_entity, substitute_concept

    unlearner = _spare(
        forget_concept=canonical_entity(task, class_key),
        overwritting_concept=substitute_concept(task),
    )

    assert unlearner._forget_caption() == expected_caption(task, f"{class_key}_0001.jpg")
    assert unlearner._overwrite_caption() == f"An image of {substitute_concept(task)}"


def test_both_sides_of_the_pair_differ_only_in_the_concept():
    """The property `caption_template` exists to guarantee."""
    unlearner = _spare(forget_concept="Mark Philippoussis", overwritting_concept="a child")

    forget = unlearner._forget_caption()
    overwrite = unlearner._overwrite_caption()

    assert forget == "An image of Mark Philippoussis"
    assert overwrite == "An image of a child"
    assert forget.replace("Mark Philippoussis", "a child") == overwrite


def test_without_a_forget_concept_the_stored_caption_is_used_unchanged():
    """Then the split's own caption is what the forget side sees, which is only correct because
    that caption is now the prompted form."""
    assert _spare()._forget_caption() is None
