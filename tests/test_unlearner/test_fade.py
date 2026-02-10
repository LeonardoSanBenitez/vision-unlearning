import pytest
import warnings
import torch
from diffusers import UNet2DConditionModel
from vision_unlearning.unlearner.fade import make_lora_elastic, ElasticLoRALinear, SharedR


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


def test_state_dict_saves_original_weights():
    """
    Verify that state_dict excludes masked weights (_weight, _bias) and keeps original names.
    """
    r_values = [4, 8, 16]
    layer = ElasticLoRALinear(32, 64, SharedR(r_values), is_lora_A=True)
    sd = layer.state_dict()
    for key in sd.keys():
        assert not key.startswith("_"), f"State dict contains internal key: {key}"


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
