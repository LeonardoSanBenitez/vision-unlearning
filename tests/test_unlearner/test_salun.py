'''Tests for the SalUn unlearner.

The centre of gravity here is the saliency mask, because that is the part of the method this
implementation does differently from the reference and the part a plausible reimplementation gets
subtly wrong. Four properties are pinned, and each one is paired with the mutation that breaks it:

1. **The threshold is a rank over EVERY parameter**, not within the trained subset. Mutation: build
   the mask from the cross-attention parameters alone. The two masks must differ.
2. **The mask keeps the LARGEST magnitudes**, not the smallest. Mutation: keep the bottom fraction.
   The two masks must differ. This one is invisible at a threshold of 0.5, where the complement of a
   half is a half, so it is tested away from 0.5.
3. **The cheap threshold construction reproduces a stable rank cut exactly**, including where
   magnitudes tie at the boundary -- which they do in practice, six figures of them on a real model.
4. **The mask multiplies the gradient BEFORE the optimizer steps.** Mutation: multiply afterwards.
   A parameter the mask excludes must not move at all.
'''
from __future__ import annotations

import os
from typing import Any, Dict

import pytest
import torch

from vision_unlearning.unlearner.salun import SALUN_MASK_FILENAME, SALUN_WEIGHTS_FILENAME, SalUn


def _unlearner(**overrides: Any) -> SalUn:
    settings: Dict[str, Any] = {
        'dataset_forget_name': 'unused/forget',
        'dataset_retain_name': 'unused/retain',
        'overwriting_concept': 'a child',
        'device': 'cpu',
    }
    settings.update(overrides)
    return SalUn(**settings)


def _magnitudes(with_ties: bool = False) -> Dict[str, torch.Tensor]:
    '''Two parameters, one of which is named like a cross-attention tensor.

    Values are deterministic and, when `with_ties` is set, quantized so that many of them collide --
    which is the only case where a value cut and a rank cut can disagree, and therefore the case that
    decides whether the cheap construction is exact.
    '''
    torch.manual_seed(11)
    plain = torch.rand(40, 40)
    cross = torch.rand(20, 20)
    if with_ties:
        plain = (plain * 8).round() / 8
        cross = (cross * 8).round() / 8
    return {'down_blocks.0.weight': plain, 'down_blocks.0.attn2.to_k.weight': cross}


def _stable_rank_mask(magnitudes: Dict[str, torch.Tensor], threshold: float) -> Dict[str, torch.Tensor]:
    '''The reference's construction, with its unstable sort made stable.

    `generate_mask.py` ranks the NEGATED magnitudes ascending and keeps `rank < cut`, which keeps the
    top fraction. Its `torch.argsort` call is unstable, so where values tie its own output is
    unspecified; `stable=True` here makes the control reproducible, which is what a control has to be.
    '''
    flat = torch.cat([tensor.flatten() for tensor in magnitudes.values()])
    cut = int(flat.numel() * threshold)
    order = torch.argsort(-flat, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel())

    masks: Dict[str, torch.Tensor] = {}
    start = 0
    for name, tensor in magnitudes.items():
        count = tensor.numel()
        masks[name] = (ranks[start:start + count] < cut).reshape(tensor.shape)
        start += count
    return masks


def _differing(left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]) -> int:
    return sum(int(torch.sum(left[name] != right[name])) for name in left)


# ---------------------------------------------------------------- parameter selection

def test_xattn_selects_only_cross_attention_parameters() -> None:
    unet = torch.nn.Module()
    unet.attn2_to_k = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    unet.other = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    module = torch.nn.Module()
    module.attn2 = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    module.attn1 = torch.nn.Linear(2, 2)  # type: ignore[assignment]

    selected = _unlearner(train_method='xattn')._select_parameter_names(module)
    assert selected == ['attn2.weight', 'attn2.bias']


def test_full_selects_every_parameter() -> None:
    module = torch.nn.Module()
    module.attn2 = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    module.attn1 = torch.nn.Linear(2, 2)  # type: ignore[assignment]

    selected = _unlearner(train_method='full')._select_parameter_names(module)
    assert set(selected) == {'attn2.weight', 'attn2.bias', 'attn1.weight', 'attn1.bias'}


def test_an_unknown_train_method_is_refused() -> None:
    module = torch.nn.Module()
    module.attn2 = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    with pytest.raises(ValueError, match='Unsupported train_method'):
        _unlearner(train_method='esd-x')._select_parameter_names(module)


def test_a_checkpoint_with_no_matching_parameters_is_refused() -> None:
    module = torch.nn.Module()
    module.attn1 = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    with pytest.raises(ValueError, match='No trainable parameters'):
        _unlearner(train_method='xattn')._select_parameter_names(module)


# ---------------------------------------------------------------- the mask construction

@pytest.mark.parametrize('with_ties', [False, True])
@pytest.mark.parametrize('threshold', [0.1, 0.3, 0.5, 0.9])
def test_the_cheap_threshold_reproduces_a_stable_rank_cut(with_ties: bool, threshold: float) -> None:
    '''The proof obligation: bisection on the bit pattern must select exactly the rank cut's mask.

    Ties are the only place the two constructions can part company, so the tied case is the one that
    decides. On a real denoiser there are six figures of them.
    '''
    magnitudes = _magnitudes(with_ties=with_ties)
    built = _unlearner(mask_threshold=threshold)._build_mask(magnitudes)
    expected = _stable_rank_mask(magnitudes, threshold)
    assert _differing(built, expected) == 0


def test_the_tied_case_is_not_vacuous() -> None:
    '''Guard on the test above: if the quantized fixture had no ties it would prove nothing.'''
    magnitudes = _magnitudes(with_ties=True)
    flat = torch.cat([tensor.flatten() for tensor in magnitudes.values()])
    assert flat.numel() - int(torch.unique(flat).numel()) > 100


def test_the_mask_keeps_the_requested_fraction() -> None:
    magnitudes = _magnitudes()
    total = sum(int(tensor.numel()) for tensor in magnitudes.values())
    built = _unlearner(mask_threshold=0.25)._build_mask(magnitudes)
    assert sum(int(torch.sum(mask)) for mask in built.values()) == int(total * 0.25)


def test_a_threshold_of_one_keeps_everything() -> None:
    magnitudes = _magnitudes()
    built = _unlearner(mask_threshold=1.0)._build_mask(magnitudes)
    assert all(bool(mask.all()) for mask in built.values())


def test_mutation_a_local_threshold_is_not_the_global_one() -> None:
    '''Mutation 1: rank within the trained subset instead of over every parameter.'''
    magnitudes = _magnitudes()
    unlearner = _unlearner(mask_threshold=0.5)
    global_mask = unlearner._build_mask(magnitudes)

    cross_only = {name: tensor for name, tensor in magnitudes.items() if 'attn2' in name}
    local_mask = unlearner._build_mask(cross_only)

    name = 'down_blocks.0.attn2.to_k.weight'
    assert int(torch.sum(global_mask[name] != local_mask[name])) > 0


def test_mutation_keeping_the_bottom_fraction_is_a_different_mask() -> None:
    '''Mutation 2: keep the smallest magnitudes instead of the largest.

    Tested at 0.25 rather than 0.5 on purpose: at a half the bottom mask is the exact complement of
    the top one, and any count-based check would pass either way.
    '''
    magnitudes = _magnitudes()
    top = _unlearner(mask_threshold=0.25)._build_mask(magnitudes)

    flat = torch.cat([tensor.flatten() for tensor in magnitudes.values()])
    cut = int(flat.numel() * 0.25)
    order = torch.argsort(flat, stable=True)  # ascending: the smallest first
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel())
    bottom: Dict[str, torch.Tensor] = {}
    start = 0
    for name, tensor in magnitudes.items():
        count = tensor.numel()
        bottom[name] = (ranks[start:start + count] < cut).reshape(tensor.shape)
        start += count

    assert _differing(top, bottom) > 0
    # And the property that makes the top mask the right one: it keeps the largest values.
    kept = torch.cat([magnitudes[name][top[name]].flatten() for name in magnitudes])
    dropped = torch.cat([magnitudes[name][~top[name]].flatten() for name in magnitudes])
    assert float(kept.min()) >= float(dropped.max())


# ---------------------------------------------------------------- the masked update

def _stepped_parameter(mask_value: bool, mask_before_step: bool) -> torch.Tensor:
    '''One Adam step on a single parameter, with the mask applied before or after the step.'''
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    mask = torch.full((4, 4), mask_value)

    optimizer.zero_grad(set_to_none=True)
    (parameter.sum() * 2.0).backward()
    if mask_before_step:
        assert parameter.grad is not None
        parameter.grad.mul_(mask)
        optimizer.step()
    else:
        optimizer.step()
        assert parameter.grad is not None
        parameter.grad.mul_(mask)
    return parameter.detach()


def test_a_parameter_the_mask_excludes_does_not_move() -> None:
    assert torch.equal(_stepped_parameter(False, mask_before_step=True), torch.ones(4, 4))


def test_a_parameter_the_mask_keeps_does_move() -> None:
    assert not torch.equal(_stepped_parameter(True, mask_before_step=True), torch.ones(4, 4))


def test_mutation_masking_after_the_step_lets_an_excluded_parameter_move() -> None:
    '''Mutation 4: apply the mask after `optimizer.step()` instead of before.

    This is the mutation that a test comparing losses would never catch, because the loss is computed
    before either ordering takes effect. Only the weights show it.
    '''
    assert not torch.equal(_stepped_parameter(False, mask_before_step=False), torch.ones(4, 4))


# ---------------------------------------------------------------- artifacts

def test_the_mask_round_trips_through_its_own_file(tmp_path: Any) -> None:
    '''The mask is saved as an artifact, not kept only in memory, so it must reload unchanged.'''
    from safetensors.torch import load_file

    magnitudes = _magnitudes()
    unlearner = _unlearner(mask_threshold=0.5, output_dir=str(tmp_path))
    mask = unlearner._build_mask(magnitudes)
    unlearner._save_mask(mask)

    reloaded = load_file(os.path.join(str(tmp_path), SALUN_MASK_FILENAME))
    assert set(reloaded) == set(mask)
    for name in mask:
        assert torch.equal(reloaded[name].bool(), mask[name])


def test_the_weight_file_carries_the_settings_that_produced_it(tmp_path: Any) -> None:
    '''The saved tensors alone do not say how they were made; the metadata has to.'''
    from safetensors import safe_open

    unlearner = _unlearner(output_dir=str(tmp_path), mask_threshold=0.3, alpha=0.7, seed=7)
    unlearner._save_weights({'down_blocks.0.attn2.to_k.weight': torch.zeros(2, 2)})

    with safe_open(os.path.join(str(tmp_path), SALUN_WEIGHTS_FILENAME), framework='pt') as handle:
        metadata = handle.metadata()
    assert metadata['mask_threshold'] == '0.3'
    assert metadata['alpha'] == '0.7'
    assert metadata['seed'] == '7'
    assert metadata['overwriting_concept'] == 'a child'


# ---------------------------------------------------------------- configuration

def test_an_unknown_mask_precision_is_refused() -> None:
    with pytest.raises(ValueError, match='Unsupported mask_precision'):
        _unlearner(mask_precision='int8')._mask_dtype()


def test_the_declared_precisions_resolve() -> None:
    assert _unlearner(mask_precision='fp32')._mask_dtype() == torch.float32
    assert _unlearner(mask_precision='bf16')._mask_dtype() == torch.bfloat16
    assert _unlearner(mask_precision='fp16')._mask_dtype() == torch.float16


def test_an_unrecognised_keyword_is_an_error() -> None:
    '''Inherited from `Unlearner`, and worth pinning here: two hyperparameters were silently
    dropped for the life of another class because pydantic ignored extras by default.'''
    with pytest.raises(Exception):
        _unlearner(mask_treshold=0.5)  # deliberate misspelling


# ---------------------------------------------------------------- the real code paths
#
# The tests above pin the mask construction, and a mutation run showed that they do NOT cover two of
# the four properties gate G6a fixes: the sign and form of the guidance term, and the ordering of the
# mask against the optimizer step. Both were tested as patterns reproduced inside the test rather
# than as behaviour of the methods, so a broken implementation passed them. The tests below drive
# `_accumulate_saliency` and `_fit` themselves, against a denoiser small enough to run on the
# processor in a second.

class _StubLatentDistribution:
    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def sample(self, generator: Any = None) -> torch.Tensor:
        return self._latents


class _StubVae(torch.nn.Module):
    '''Encodes deterministically, so a test's images map to fixed latents.'''

    def __init__(self, latent_size: int) -> None:
        super().__init__()
        self.config = type('Config', (), {'scaling_factor': 1.0})()
        self._latent_size = latent_size

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def encode(self, pixel_values: torch.Tensor) -> Any:
        batch = pixel_values.shape[0]
        generator = torch.Generator().manual_seed(3)
        latents = torch.randn(batch, 4, self._latent_size, self._latent_size, generator=generator)
        return type('Encoded', (), {'latent_dist': _StubLatentDistribution(latents)})()


class _StubTextEncoder(torch.nn.Module):
    '''Maps token ids to embeddings deterministically, so two captions differ but are reproducible.'''

    def __init__(self, cross_dim: int) -> None:
        super().__init__()
        self._cross_dim = cross_dim

    def forward(self, input_ids: torch.Tensor) -> Any:
        seed = int(input_ids.sum()) % 10000
        generator = torch.Generator().manual_seed(seed)
        embeddings = torch.randn(input_ids.shape[0], 7, self._cross_dim, generator=generator)
        return (embeddings,)


class _StubTokenizer:
    model_max_length = 7

    def __call__(self, texts: Any, **keywords: Any) -> Any:
        ids = torch.zeros(len(texts), self.model_max_length, dtype=torch.long)
        return type('Tokens', (), {'input_ids': ids})()


class _StubPipeline:
    def __init__(self, unet: torch.nn.Module, latent_size: int, cross_dim: int) -> None:
        from diffusers import DDPMScheduler

        self.unet = unet
        self.vae = _StubVae(latent_size)
        self.text_encoder = _StubTextEncoder(cross_dim)
        self.tokenizer = _StubTokenizer()
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)


def _tiny_unet(latent_size: int, cross_dim: int) -> torch.nn.Module:
    from diffusers import UNet2DConditionModel

    torch.manual_seed(5)
    return UNet2DConditionModel(
        sample_size=latent_size,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        cross_attention_dim=cross_dim,
        attention_head_dim=4,
        norm_num_groups=8,
    )


def _batch(batch_size: int = 1) -> Dict[str, torch.Tensor]:
    return {
        'pixel_values': torch.zeros(batch_size, 3, 16, 16),
        'input_ids': torch.full((batch_size, 7), 4, dtype=torch.long),
        'forget_ids': torch.full((batch_size, 7), 9, dtype=torch.long),
    }


def test_the_saliency_gradient_matches_an_undecomposed_backward() -> None:
    '''Pins the mask objective: its sign, its guidance form, and the decomposition that computes it.

    `_accumulate_saliency` splits one backward into two so that only one activation graph is ever
    alive. Here the objective is built directly instead -- both forward passes differentiated, one
    `backward()` -- and the two gradients are compared. Any change to the sign or the shape of
    `(1 + g) * F - g * N` moves one and not the other.
    '''
    from diffusers import DDPMScheduler

    latent_size, cross_dim = 8, 16
    unlearner = _unlearner(device='cpu', mask_guidance=7.5)

    unet = _tiny_unet(latent_size, cross_dim)
    pipeline = _StubPipeline(unet, latent_size, cross_dim)
    accumulated = unlearner._accumulate_saliency(pipeline, [_batch()])

    reference_unet = _tiny_unet(latent_size, cross_dim)
    reference_pipeline = _StubPipeline(reference_unet, latent_size, cross_dim)
    for parameter in reference_unet.parameters():
        parameter.requires_grad_(True)
    scheduler = DDPMScheduler.from_config(reference_pipeline.scheduler.config)
    generator = torch.Generator(device='cpu').manual_seed(unlearner.seed)
    batch = _batch()
    _, noise, timesteps, noisy = unlearner._noised_latents(reference_pipeline, batch, generator, scheduler)
    conditioned = unlearner._encode(reference_pipeline, batch['input_ids'])
    unconditioned = unlearner._encode_empty(reference_pipeline, batch['input_ids'].shape[0])

    reference_unet.zero_grad(set_to_none=True)
    forget_out = reference_unet(noisy, timesteps, encoder_hidden_states=conditioned).sample
    null_out = reference_unet(noisy, timesteps, encoder_hidden_states=unconditioned).sample
    predicted = (1.0 + unlearner.mask_guidance) * forget_out - unlearner.mask_guidance * null_out
    (-torch.nn.functional.mse_loss(noise, predicted)).backward()

    largest = 0.0
    scale = 0.0
    for name, parameter in reference_unet.named_parameters():
        assert parameter.grad is not None
        expected = parameter.grad.detach().abs()
        largest = max(largest, float(torch.max(torch.abs(accumulated[name] - expected))))
        scale = max(scale, float(torch.max(expected)))
    assert scale > 0, 'the reference gradient is identically zero, so this proves nothing'
    assert largest / scale < 1e-4, f'relative difference {largest / scale}'


def test_a_masked_out_parameter_does_not_move_during_a_real_fit() -> None:
    '''Pins the ordering: `_fit` must multiply the gradient by the mask BEFORE stepping.

    Driving `_fit` rather than reproducing its shape is the whole point -- the earlier version of
    this check reproduced the pattern inside the test and passed against an implementation that
    masked after the step.
    '''
    latent_size, cross_dim = 8, 16
    unet = _tiny_unet(latent_size, cross_dim)
    pipeline = _StubPipeline(unet, latent_size, cross_dim)
    unlearner = _unlearner(device='cpu', train_method='xattn', num_train_epochs=1, learning_rate=0.1)

    trained_names = unlearner._select_parameter_names(unet)
    frozen_name = trained_names[0]
    mask = {name: torch.ones_like(unet.get_parameter(name), dtype=torch.bool) for name in trained_names}
    mask[frozen_name] = torch.zeros_like(unet.get_parameter(frozen_name), dtype=torch.bool)
    before = {name: unet.get_parameter(name).detach().clone() for name in trained_names}

    trained = unlearner._fit(pipeline, mask, [_batch()], [_batch()])

    assert torch.equal(trained[frozen_name], before[frozen_name].cpu()), \
        'a parameter the mask excludes moved, so the mask was applied after the step'
    moved = [name for name in trained_names if not torch.equal(trained[name], before[name].cpu())]
    assert moved, 'nothing moved at all, so the test cannot distinguish the two orderings'
