'''The one implementation of this ablation's CLIP scoring, shared by every script that needs it.

`clip_diff_campaign.py` (the ten selected entities, both seeds, thirteen checkpoints) and
`random_ten_control.py` (ten entities nobody selected, one checkpoint) ask the same question of
different image sets, so the arithmetic lives here once instead of twice. Everything below is a thin
arrangement of the library's own `MetricImageTextSimilarity(metrics=['clip'])` -- the primitive
`pipeline_06` and the every-epoch ablation use, so the numbers sit on the same scale as the existing
Stable Diffusion 1.4 curves. Nothing here re-implements a metric.

Two text conditions are scored for every image, never one:

* `clip_diff = clip_on - clip_off` against the entity's OWN prompt ("An image of Mark Philippoussis").
  This is the canonical interference metric; more negative means the image agrees less with the prompt.
* `clip_overwrite_diff` against the OVERWRITE concept the trainer distilled toward ("An image of a
  child"). SPARE does not merely degrade the target, it moves it toward a replacement concept, so this
  is the direction the training objective actually pushes in. It is a diagnostic, reported alongside
  `clip_diff` and never substituted for it.

Scoring both is what makes the metric question answerable: if an image visibly changes while
`clip_diff` stays flat, the second column says whether the signal went somewhere else or whether CLIP
saw nothing at all.
'''
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

TASK: Literal['people'] = 'people'
METHOD: Literal['distil'] = 'distil'
MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'

_HERE = Path(__file__).resolve().parent
SELECTION = _HERE.parent / 'every_epoch' / 'assets' / 'selection_people.json'


def epoch_sort_key(epoch: Any) -> Any:
    '''Sort key placing the off-baseline (`epoch: null`) before every trained checkpoint.'''
    return (epoch is not None, epoch)


def prompts_for(entities: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    '''The own-prompt and overwrite-prompt of each entity, from the helper generation itself used.

    @param entities: entity names in metadata form (underscores), e.g. `Mark_Philippoussis`.
    @return: (own prompt by entity, overwrite prompt by entity). Never retyped anywhere else.
    '''
    from vision_unlearning.datasets.testbed import get_target_overwrite

    own: Dict[str, str] = {}
    overwrite: Dict[str, str] = {}
    for name in entities:
        display, overwrite_concept = get_target_overwrite(TASK, METHOD, name)
        own[name] = f"An image of {display}"
        overwrite[name] = f"An image of {overwrite_concept}"
    return own, overwrite


def score_entities(
    manifest_rows: List[Dict[str, Any]],
    entities: List[str],
) -> Tuple[Dict[str, Any], List[int], int]:
    '''Scores every image of `entities` in a campaign-shaped manifest, both text conditions.

    A campaign-shaped manifest is a list of rows `{"epoch": n_or_null, "entity": ..., "path": ...}`,
    with `epoch: null` marking the off-baseline the differences are taken against. Every entity must
    hold exactly one row per epoch value present in the manifest; a missing image raises rather than
    being silently dropped, because a trajectory with a hole in it is not a trajectory.

    @param manifest_rows: the rows read back from a manifest written at generation time.
    @param entities: the entities to score, in the order the caller wants them reported.
    @return: (per-entity results, the trained epochs in ascending order, images scored).
    '''
    from PIL import Image

    from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity

    own_prompt, overwrite_prompt = prompts_for(entities)

    by_entity: Dict[str, Dict[Any, str]] = {}
    for row in manifest_rows:
        by_entity.setdefault(row['entity'], {})[row['epoch']] = row['path']

    missing = sorted(set(entities) - set(by_entity))
    assert not missing, f"manifest is missing entities: {missing}"

    epochs = sorted({row['epoch'] for row in manifest_rows}, key=epoch_sort_key)
    assert epochs[0] is None, "manifest has no off-baseline row (epoch null)"
    trained_epochs: List[int] = [epoch for epoch in epochs if epoch is not None]

    metric = MetricImageTextSimilarity(metrics=['clip'])
    per_entity: Dict[str, Any] = {}
    n_images_scored = 0
    for name in entities:
        ordered: List[Any] = [None] + list(trained_epochs)
        for epoch in ordered:
            assert epoch in by_entity[name], f"{name} has no image at epoch {epoch}"
        images: List[Any] = [Image.open(by_entity[name][epoch]).convert('RGB') for epoch in ordered]
        n_images_scored += len(images)
        own = [float(s['clip']) for s in metric.score_batch_same_text(images, own_prompt[name])]
        over = [float(s['clip']) for s in metric.score_batch_same_text(images, overwrite_prompt[name])]
        clip_off, clip_off_overwrite = own[0], over[0]
        per_entity[name] = {
            'clip_off': clip_off,
            'clip_off_overwrite': clip_off_overwrite,
            'trajectory': [
                {
                    'epoch': epoch,
                    'path': by_entity[name][epoch],
                    'clip_on': own[index + 1],
                    'clip_diff': own[index + 1] - clip_off,
                    'clip_on_overwrite': over[index + 1],
                    'clip_overwrite_diff': over[index + 1] - clip_off_overwrite,
                }
                for index, epoch in enumerate(trained_epochs)
            ],
        }
    return per_entity, trained_epochs, n_images_scored
