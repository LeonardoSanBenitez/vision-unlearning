"""Baseline competence: own score, rank and specificity per image, the screen and ``f_min`` per task.

Fails on the code before this change: ``baseline_competence`` does not exist, so nothing produced the
table the screen and the scale reference of equalization are computed from.
"""
import json
import os
from typing import Any, Dict, List, Literal

import pytest

from vision_unlearning.benchmarks.I_care import baseline_competence as bc
from vision_unlearning.benchmarks.I_care.equalization import IntegrityFailure
from vision_unlearning.benchmarks.I_care.provenance import prompt_list_sha256
from vision_unlearning.datasets.entity_names import generation_prompt
from vision_unlearning.datasets.testbed import get_generated_dataset_file, get_shared_baseline_folder

TASK: Literal['people'] = 'people'
ENTITIES = ['A_One', 'B_Two', 'C_Three']
SEEDS = [42, 43]
SEEDS4 = [42, 43, 44, 45]


def test_own_score_rank_and_specificity() -> None:
    scores = {
        'A_One': {42: [30.0, 20.0, 10.0], 43: [20.0, 25.0, 15.0]},
        'B_Two': {42: [10.0, 30.0, 20.0], 43: [10.0, 30.0, 30.0]},
        'C_Three': {42: [10.0, 10.0, 31.0], 43: [11.0, 12.0, 32.0]},
    }
    out = bc.summarise(ENTITIES, SEEDS, scores)
    a42, a43 = out['per_image']['A_One']['42'], out['per_image']['A_One']['43']
    assert (a42['own'], a42['rank'], a42['specificity']) == (30.0, 1, 15.0)
    assert (a43['own'], a43['rank'], a43['specificity']) == (20.0, 2, 0.0)
    # A tie counts against the entity: B at seed 43 shares its score with C's prompt.
    assert out['per_image']['B_Two']['43']['rank'] == 2
    specificities = [15.0, 0.0, 15.0, 10.0, 21.0, 20.5]
    assert out['mean_specificity'] == pytest.approx(sum(specificities) / 6)
    assert out['f_min'] == pytest.approx(0.5 * sum(specificities) / 6)
    assert out['screen']['entities'] == 3


def test_a_missing_score_is_an_integrity_failure() -> None:
    scores = {e: {s: [1.0, 2.0, 3.0] for s in SEEDS} for e in ENTITIES}
    scores['B_Two'][43] = [1.0, 2.0]
    with pytest.raises(IntegrityFailure, match='B_Two at seed 43: 2 scores for 3 prompts'):
        bc.summarise(ENTITIES, SEEDS, scores)


def test_a_scale_reference_that_is_not_positive_raises() -> None:
    flat = {e: {s: [5.0, 5.0, 5.0] for s in SEEDS} for e in ENTITIES}
    with pytest.raises(IntegrityFailure):
        bc.summarise(ENTITIES, SEEDS, flat)


class FakeScorer:
    """Scores an image against a prompt by whether the image's file was generated from that prompt."""

    def __init__(self) -> None:
        self.calls: List[Any] = []

    def score_batch_same_text(self, images: List[Any], text: str) -> List[Dict[str, float]]:
        names = [os.path.basename(image.filename) for image in images]
        self.calls.append((names, text))
        return [{'clip': 30.0 + int(name.split('_')[1]) - 42 if text in name else 10.0} for name in names]


def _baseline(base: str, seeds: List[int]) -> None:
    from PIL import Image
    prompts = [generation_prompt(TASK, e) for e in ENTITIES]  # type: ignore[arg-type]
    with open(os.path.join(base, f'metadata_{TASK}_2_enriched_filtered.json'), 'w', encoding='utf-8') as handle:
        json.dump([{'name': e} for e in ENTITIES], handle)
    folder = get_shared_baseline_folder(TASK, base)
    os.makedirs(folder)
    for seed in seeds:
        for prompt in prompts:
            Image.new('RGB', (4, 4)).save(os.path.join(folder, get_generated_dataset_file('off', seed, prompt)))
    with open(os.path.join(folder, 'provenance.json'), 'w', encoding='utf-8') as handle:
        json.dump({'seeds': seeds, 'prompts_ordered': prompts, 'prompt_list_sha256': prompt_list_sha256(prompts)}, handle)


def test_scored_as_pipeline_06_scores_one_entity_per_call_and_written(tmp_path: Any) -> None:
    base = str(tmp_path)
    _baseline(base, SEEDS4)
    scorer = FakeScorer()
    record = bc.write_competence(TASK, base, seeds=SEEDS4, scorer=scorer)  # type: ignore[arg-type]
    prompts = [generation_prompt(TASK, e) for e in ENTITIES]  # type: ignore[arg-type]
    # One call per (entity, prompt), each holding that entity's seed images in seed order.
    assert len(scorer.calls) == len(ENTITIES) ** 2
    for k, (names, text) in enumerate(scorer.calls):
        own = prompts[k // len(ENTITIES)]
        assert names == [get_generated_dataset_file('off', s, own) for s in SEEDS4]
        assert text == prompts[k % len(ENTITIES)]
    with open(bc.competence_path(TASK, base), encoding='utf-8') as handle:
        written = json.load(handle)
    assert written == json.loads(json.dumps(record))
    assert written['scores']['B_Two']['43'] == [10.0, 31.0, 10.0]
    assert written['per_image']['B_Two']['43'] == {'own': 31.0, 'rank': 1, 'specificity': 21.0}
    assert written['prompt_list_sha256'] == prompt_list_sha256(prompts)
    assert written['screen']['recognised'] == 3 and written['seeds'] == SEEDS4


def test_a_baseline_recorded_with_other_seeds_is_refused(tmp_path: Any) -> None:
    base = str(tmp_path)
    _baseline(base, SEEDS)
    with pytest.raises(ValueError):
        bc.write_competence(TASK, base, seeds=[42, 44], scorer=FakeScorer())  # type: ignore[arg-type]
    assert not os.path.exists(bc.competence_path(TASK, base))
