"""Pairwise entity similarity for the I-CARE benchmark.

Holds the :class:`Similarity` artifact — the canonical ``similarity_{s}_{task}.json`` for one
(task, similarity metric), owning the per-metric computation (``jacc`` inline with a ``.partial``
checkpoint, ``dino`` from the method-agnostic baseline DINOv2 embeddings, ``act`` from the
cross-attention fingerprints) and the shared local -> HuggingFace -> from-scratch storage cascade —
together with the ``jacc_metric_score`` attribute-overlap helper it uses for the ``jacc`` metric.

``ResultTemplateSimilarityMatrix`` (in ``result_templates.py``) is a thin display reader over this
artifact; this module intentionally does not depend on ``result_templates`` so there is no import
cycle.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

from vision_unlearning.artifact import SingleFileArtifact
from vision_unlearning.datasets.testbed import get_metadata_filtered, get_target_overwrite
from vision_unlearning.benchmarks.I_care.configuration import (
    type_task,
    type_model,
    type_s,
)


def jacc_metric_score(entity_1: str, entity_2: str, metadata_filtered: List[Dict[str, Any]], entity_col: str = 'name') -> float:
    """
    Jaccard similarity between two entities, based on their attributes.
    Each attribute (column) contributes between 0 and 1 to the similarity
    We do not know the types and ranges of the attributes beforehand.
    For each attribute, both values for the two entities must be non-NaN and of the same type, otherwise we ignore that attribute (contribution 0).
    The calculation for each attribute is as follows:
    * If the attribute is categorical (str or bool), the contribution is 1 if the two entities have the same value for that attribute, and 0 otherwise.
    * If the attribute is numerical, and both values are between 0 and 1, the contribution is 1 - abs(value_1 - value_2)
    * If the attribute is numerical, and both values are between 1 and 100, the contribution is 1 - abs(value_1 - value_2) / 100
    * else, the contribution is 0 (we do not know how to handle it, so we ignore it)
    """
    # Get the rows corresponding to the two entities
    row_1 = next((row for row in metadata_filtered if row[entity_col] == entity_1), None)
    row_2 = next((row for row in metadata_filtered if row[entity_col] == entity_2), None)
    if row_1 is None or row_2 is None:
        raise ValueError(f"Entities {entity_1} and/or {entity_2} not found in metadata")
    if set(row_1.keys()) != set(row_2.keys()):
        raise ValueError(f"Entities {entity_1} and {entity_2} must have the same attributes")

    # Calculate similarity for each attribute
    similarity = 0.0
    valid_attributes = 0
    for attr in row_1.keys():
        value_1 = row_1[attr]
        value_2 = row_2[attr]

        if pd.isna(value_1) or pd.isna(value_2) or type(value_1) != type(value_2):
            continue  # ignore this attribute

        if isinstance(value_1, (str, bool)):
            similarity += 1.0 if value_1 == value_2 else 0.0
            valid_attributes += 1
        elif isinstance(value_1, (int, float)):
            if 0 <= value_1 <= 1 and 0 <= value_2 <= 1:
                similarity += 1 - abs(value_1 - value_2)
                valid_attributes += 1
            elif 1 < value_1 <= 100 and 1 < value_2 <= 100:
                similarity += 1 - abs(value_1 - value_2) / 100
                valid_attributes += 1
            else:
                continue  # ignore this attribute
        else:
            continue  # ignore this attribute
    similarity = similarity / valid_attributes if valid_attributes > 0 else 0.0

    # Post checks
    assert valid_attributes > 0, f"Expected at least one valid attribute for entities {entity_1} and {entity_2}, got {valid_attributes}."
    assert type(similarity) == float
    assert 0 <= similarity <= 1
    return similarity


class Similarity(SingleFileArtifact):
    """Pairwise 100x100 similarity matrix for one (task, similarity_metric).

    Backs the canonical ``similarity_{s}_{task}.json`` (appendix ``ap:metadata``). This is
    where the heavy per-metric computation lives — ``jacc`` inline with a ``.partial``
    checkpoint, ``dino`` from the method-agnostic baseline DINOv2 embeddings, ``act`` from the
    cross-attention fingerprints; ``ResultTemplateSimilarityMatrix`` is a thin reader over it.

    The result is a list of 100 emitter-row records
    ``{"emitter": <name>, <receiver_name>: <float>, ...}``, indexed by the metadataFiltered
    ``name`` field (same keys the SimilarityMatrix result template has always used).
    """
    model: type_model = 'sd1.4'
    task: type_task = 'people'
    similarity_metric: type_s = 'clip'

    def _get_data_path_remote(self) -> str:
        return f"similarity_{self.similarity_metric}_{self.task}.json"

    def _get_partial_path_local(self) -> str:
        return self._get_data_path_local() + '.partial'

    def _validate(self, data: Any) -> None:
        assert isinstance(data, list)
        assert all(isinstance(row, dict) and 'emitter' in row for row in data)

    def _compute_from_scratch(self) -> List[Dict[str, Any]]:
        metadata_filtered: List[Dict[str, Any]] = get_metadata_filtered(self.task)
        labels: List[str] = [e['name'] for e in metadata_filtered]

        if self.similarity_metric == 'clip':
            raise NotImplementedError(
                "CLIP similarity matrix not found locally or in HuggingFace Hub. "
                "Recomputing from scratch requires a GPU with CLIP/SD 1.4 and is not "
                "supported here.  Either upload the precomputed matrix to HF or "
                "run pipeline_02 with --similarity clip on a GPU machine."
            )

        elif self.similarity_metric == 'jacc':
            # The partial checkpoint lives next to the canonical file.
            os.makedirs(os.path.dirname(self._get_partial_path_local()), exist_ok=True)

            # 100x100 matrix; resume from a partial checkpoint if present.
            if os.path.exists(self._get_partial_path_local()) and not self.recompute_if_exists:
                df_similarities = pd.read_json(self._get_partial_path_local(), orient='records')
                df_similarities.set_index('emitter', inplace=True)
                assert df_similarities.index.to_list() == labels
            else:
                df_similarities = pd.DataFrame(index=labels, columns=labels)

            for entity_emitter, row_emitter in df_similarities.iterrows():
                for entity_receiver in row_emitter.index:
                    if pd.isna(df_similarities.loc[entity_emitter, entity_receiver]):  # type: ignore
                        similarity: float = jacc_metric_score(str(entity_emitter), str(entity_receiver), metadata_filtered)
                        df_similarities.loc[entity_emitter, entity_receiver] = similarity

                # Checkpoint at the end of each row
                df_similarities.reset_index(names='emitter').to_json(self._get_partial_path_local(), orient='records')

        elif self.similarity_metric == 'dino':
            from collections import defaultdict
            # Baseline embeddings are produced by the original model and carry no method or
            # epoch (the per-method baseline name is an obsolete artifact).
            embedding_path = os.path.join(
                self.base_folder, 'datasets',
                f'embeddings_{self.task}_original.json'
            )
            assert os.path.exists(embedding_path), (
                f"Baseline DINOv2 embeddings not found at {embedding_path}. "
                f"Run pipeline_05 (compute embeddings) for the baseline first."
            )
            with open(embedding_path, encoding='utf-8') as f:
                raw = json.load(f)

            # Build forward mapping: metadata_name -> expected prompt string.
            # The embedding file's 'prompt' field is "An image of {get_target_overwrite(...)[0]}"
            # and is consistent across tasks.  We key by prompt rather than 'prompted_entity'
            # because 'prompted_entity' has inconsistent formatting across tasks, whereas
            # 'prompt' is clean.  get_target_overwrite's `method` parameter is ignored by the
            # transform, so any method value produces the same result.
            ent_list = [e['name'] for e in metadata_filtered]
            meta_to_prompt: Dict[str, str] = {
                name: f"An image of {get_target_overwrite(self.task, 'distil', name)[0]}"
                for name in ent_list
            }

            # Group embeddings by prompt, compute mean unit-vector per entity
            buckets: Dict[str, List[List[float]]] = defaultdict(list)
            for entry in raw['embeddings']:
                buckets[entry['prompt']].append(entry['embedding'])

            entity_embeddings: Dict[str, np.ndarray] = {}
            for meta_name in ent_list:
                expected_prompt = meta_to_prompt[meta_name]
                if expected_prompt not in buckets:
                    raise ValueError(
                        f"No embeddings found for '{meta_name}' "
                        f"(expected prompt: '{expected_prompt}'). "
                        f"First 3 available prompts: {list(buckets.keys())[:3]}"
                    )
                vecs = buckets[expected_prompt]
                arr = np.array(vecs)
                mean_vec = arr.mean(axis=0)
                entity_embeddings[meta_name] = mean_vec / np.linalg.norm(mean_vec)

            # Build N×N cosine similarity matrix (dot product of unit vectors)
            mat = np.array([entity_embeddings[e] for e in ent_list])
            sim_matrix = mat @ mat.T
            df_similarities = pd.DataFrame(sim_matrix, index=ent_list, columns=ent_list)

        elif self.similarity_metric == 'act':
            from vision_unlearning.utils.mechanistic_interpretability import (
                load_act_fingerprints,
                compute_cosine_similarity_matrix,
            )
            fingerprints = load_act_fingerprints(
                task=self.task,
                model=self.model,
                base_folder=self.base_folder,
            )
            df_similarities = compute_cosine_similarity_matrix(fingerprints, labels)

        else:
            raise NotImplementedError(
                f"Unsupported similarity_metric={self.similarity_metric!r}"
            )

        return cast(
            List[Dict[str, Any]],
            df_similarities.reset_index(names='emitter').to_dict(orient='records'),
        )

    def compute(self) -> List[Dict[str, Any]]:
        """Resolve the pairwise similarity matrix from local disk, HuggingFace, or compute it."""
        return cast(List[Dict[str, Any]], self._resolve())
