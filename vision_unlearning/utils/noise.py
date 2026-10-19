'''The initial noise of a generated image: where it is drawn, from which generator, and its hash.

A diffusion pipeline turns an initial latent into an image. Two images can only be compared pixel by
pixel, or embedded and compared, as "the same picture under two models" if they started from the same
initial latent. This module is the one place that decides how that latent is drawn, so every route
that generates or captures an image draws it the same way.

Two conventions exist, named by `type_noise`:

* ``'stream'`` -- the historical one, and the default everywhere: one generator per seed on the
  generation device, advanced across the whole prompt list in order. An image's noise then depends on
  its seed, on its position in the prompt list, on the batch size and on the graphics card model,
  because a card's random-number generator draws differently for different batch shapes and different
  hardware. Nothing in this module implements it; the callers keep their existing code for it.
* ``'per_image_processor'`` -- every image is given its own generator, created on the processor and
  seeded to the image's seed, and its latent is drawn from that generator alone. An image's noise then
  depends on its seed and nothing else: not on the prompt, not on its position, not on the batch it was
  generated in, and not on the card. Every entity generated at one seed starts from the same latent.

Under the second convention the latents are built here and handed to the pipeline as ``latents=``,
rather than handing the pipeline the generators and letting it draw. The values are identical either
way -- `diffusers` draws with the same `randn_tensor` call -- but building them here is what makes it
possible to hash the exact tensor the pipeline receives, before the pipeline scales it by the
scheduler's ``init_noise_sigma``. A recorded hash is therefore a statement about the input of the
generation, not about its output, and two images whose hashes agree started from the same noise.

What the hashes do not prove: that the tensor was consumed. A caller that records the hash of one
tensor and passes another would record a true hash of the wrong thing; the tests of every producer
drive a recording stand-in pipeline that re-hashes what it receives, which is the check for that.

``torch`` and ``diffusers`` are imported inside the functions, so the hash encoding can be read and
documented without them.
'''
import hashlib
from typing import Any, List, Literal, Optional, Tuple


type_noise = Literal['stream', 'per_image_processor']

# Bumped whenever the bytes fed to the hash change, so a recorded hash names the encoding it was
# computed under and a reader can refuse one it does not understand instead of comparing unlike
# strings.
LATENT_HASH_SCHEMA_VERSION = 1


def generation_generators(seed: int, n: int) -> List[Any]:
    '''`n` independent processor generators, each seeded to `seed`.

    Independent objects rather than one generator repeated: drawing from one advances it, so a list
    holding the same object `n` times would hand the second image the second draw.
    '''
    import torch
    return [torch.Generator(device='cpu').manual_seed(seed) for _ in range(n)]


def latent_sha256(latent: Any) -> str:
    '''The hash of one image's initial latent, versioned.

    Computed over the schema version, the dtype, the shape and the tensor's bytes in C order, so two
    tensors with equal bytes but a different dtype or shape never share a hash. The tensor is copied
    to the processor first; the bytes do not depend on which device held it.
    '''
    import torch
    tensor = latent.detach().to('cpu').contiguous()
    header = f'v{LATENT_HASH_SCHEMA_VERSION}|{tensor.dtype}|{tuple(tensor.shape)}|'.encode('ascii')
    payload = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + payload).hexdigest()


def _latent_dtype(pipeline: Any) -> Any:
    '''The dtype a `diffusers` text-to-image pipeline draws its latents in.

    The pipelines draw in the dtype of the prompt embeddings, which is the text encoder's dtype, or
    the denoiser's when there is no text encoder.
    '''
    text_encoder = getattr(pipeline, 'text_encoder', None)
    if text_encoder is not None:
        return text_encoder.dtype
    return pipeline.unet.dtype


def initial_latents(
    pipeline: Any,
    seed: int,
    n: int,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Tuple[Any, List[str]]:
    '''The initial latents of `n` images at one seed, under ``'per_image_processor'``, and their hashes.

    Each image's latent is drawn on the processor by its own generator seeded to `seed`, with
    `diffusers.utils.torch_utils.randn_tensor` at the pipeline's latent shape and dtype -- the exact
    call the pipeline would make itself if handed `generation_generators(seed, n)`. All `n` latents
    are therefore equal, and equal to the latent of a batch of one at the same seed.

    Returns the stacked batch, still on the processor (the pipeline moves it to its own device and
    scales it), and one hash per image, computed before any scaling.

    @param height: image height in pixels. None means the pipeline's own default,
        ``unet.config.sample_size * vae_scale_factor`` (512 for Stable Diffusion 1.x). Give both or
        neither.
    @param width: image width in pixels. See height.
    '''
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    if (height is None) != (width is None):
        raise ValueError(f'height and width must be given together; got height={height}, width={width}')
    if n < 1:
        raise ValueError(f'n must be at least 1, got {n}')
    scale = pipeline.vae_scale_factor
    if height is None or width is None:
        height = width = pipeline.unet.config.sample_size * scale
    shape = (n, pipeline.unet.config.in_channels, height // scale, width // scale)
    latents = randn_tensor(
        shape,
        generator=generation_generators(seed, n),
        device=torch.device('cpu'),
        dtype=_latent_dtype(pipeline),
    )
    return latents, [latent_sha256(latents[i]) for i in range(n)]
