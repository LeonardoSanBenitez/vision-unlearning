from typing import Literal, Tuple, List, Dict, Optional, Any
import json
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vision_unlearning.metrics.text_and_text import MetricTextTextSimilarity
from vision_unlearning.utils.logger import get_logger


logger = get_logger('testbed')


##########################################
# Target handling
##########################################
def get_target_preprocessed(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    target: str,
) -> str:
    target_preprocessed: str
    # TODO THIS SHOULD FOLLOW THE RULES CURRENTLY CODEDE AT get_target_overwrite!!!!!!!!!!
    if task == 'people':
        target_preprocessed = target  # TODO
    elif task == 'breeds':
        target_preprocessed = target  # TODO
    elif task == 'scenes':
        article = 'an' if (target[0].lower() in 'aeiou') else 'a'
        target_preprocessed = f"{article} {target} scene"
    else:
        raise NotImplementedError()

    return target_preprocessed


def get_target_overwrite(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    target: str,
) -> Tuple[str, str]:
    '''
    @return preprocessed target, target_overwrite
    '''
    # TODO THIS SHOULD USE  get_target_preprocessed FOR THE TARGET!!!!
    if task == 'people':
        # target does NOT need to have an article,for example: picture of brad pitt

        # target_race = metadata_filtered[index]['race'].replace('indian_middleEastern_latinoHispanic', 'middle eastern') # enum: white, asian, black, indian_middleEastern_latinoHispanic
        # target_gender = 'male' if metadata_filtered[index]['gender']=='M' else 'female'  # Enum[M, F]
        # article = 'an' if (target_race[0].lower() in 'aeiou') else 'a'
        # target_overwrite = f"{article} {target_race} {target_gender}"  # For munba this is only the retain concept for final evaluation, there is no overwriting
        target_overwrite = 'a child'
    elif task == 'breeds':
        # target does needs to have an article,for example: picture of a poodle
        target_overwrite = 'a cat'
        article = 'an' if (target[0].lower() in 'aeiou') else 'a'
        target = re.sub(r'\bdog\b', '', target, flags=re.IGNORECASE)
        target = f"{article} {target}"
    elif task == 'scenes':
        # target does needs to have an article,for example: picture of a phone_booth
        target_overwrite = 'the moon'
        article = 'an' if (target[0].lower() in 'aeiou') else 'a'
        target = f"{article} {target} scene"

    else:
        raise NotImplementedError()

    target = target.replace('_', ' ')
    target = re.sub(r'\s+', ' ', target).strip()

    assert isinstance(target_overwrite, str)
    assert isinstance(target, str)
    assert len(target) >= 3

    return target, target_overwrite


##########################################
# Model and generated data
##########################################
def get_generated_dataset_folder(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    target: str,
) -> str:
    # By convention, I'm passing here the preprocessed target... TODO CHANGE
    return f'assets/datasets/generated_{task}_{target}_{method}_{num_train_epochs:03d}'


def get_generated_dataset_file(
    lora_state: Literal['on', 'off'],
    seed: int,
    prompt: str,
) -> str:
    return f'{lora_state}_{seed:02}_{prompt}.png'


def get_unlearned_model_folder(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    target: str,
) -> str:
    # By convention, I'm passing here the NON preprocessed target
    return f"assets/models/{task}_{target}_{method}_{num_train_epochs:03d}"


def exists_unlearned_model(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    target: str,
) -> bool:
    model_path = get_unlearned_model_folder(task, method, num_train_epochs, target)
    if method == 'uce':
        return os.path.exists(os.path.join(model_path, 'uce_sd_weights.safetensors'))
    else:
        return os.path.exists(os.path.join(model_path, 'pytorch_lora_weights.safetensors'))


def exists_unlearned_dataset(generated_dataset_output_path: str, generate_dataset_seeds: List[int], prompts: List[str]) -> bool:
    if not os.path.exists(generated_dataset_output_path):
        return False
    if len(os.listdir(generated_dataset_output_path)) != len(generate_dataset_seeds) * len(['on', 'off']) * len(prompts) + 1:
        return False
    if not all(filename.endswith('.png') or filename.endswith('.jsonl') for filename in os.listdir(generated_dataset_output_path)):
        return False
    return True


##########################################
# Metadata files
##########################################
def get_metadata_filtered(
    task: Literal['scenes', 'objects', 'breeds', 'people']
) -> List[Dict[str, Any]]:
    with open(f"assets/metadata_{task}_2_enriched_filtered.json", "r", encoding="utf-8") as f:
        metadata_filtered = json.load(f)
    assert isinstance(metadata_filtered, list)
    return metadata_filtered


##########################################
# Similarity
##########################################
def get_similarity_clip_df(task: Literal['scenes', 'objects', 'breeds', 'people']) -> pd.DataFrame:
    df_similarities_clip = pd.read_json(f'assets/similarity_clip_{task}.json', orient='records')
    df_similarities_clip.set_index('emitter', inplace=True)
    return df_similarities_clip


def calculate_similarity_clip(task: Literal['scenes', 'objects', 'breeds', 'people'], labels: List[str]):
    clip_text_metric = MetricTextTextSimilarity(metrics=['clip_text'])

    # Load existing
    if os.path.exists(f'assets/similarity_clip_{task}.json'):
        df_similarities_clip = get_similarity_clip_df(task)
        assert df_similarities_clip.index.to_list() == labels
    else:
        df_similarities_clip = pd.DataFrame(index=labels, columns=labels)

    # Calculate
    for entity_emitter, row_emitter in df_similarities_clip.iterrows():
        # break
        print(f'Analying similarities for entity_emitter={entity_emitter}')
        for entity_receiver in row_emitter.index:
            if pd.isna(df_similarities_clip.loc[entity_emitter, entity_receiver]):  # type: ignore
                similarity: float = clip_text_metric.score(
                    get_target_preprocessed(task, str(entity_emitter)),
                    get_target_preprocessed(task, str(entity_receiver)),
                )['clip_text']
                df_similarities_clip.loc[entity_emitter, entity_receiver] = similarity

        # Save at the end of each row
        df_similarities_clip.reset_index(names='emitter').to_json(f'assets/similarity_clip_{task}.json', orient='records')

    return df_similarities_clip


def plot_heatmap(df, figsize=None, cmap="viridis", title="Heatmap"):
    """
    Plot a heatmap for a square DataFrame with all labels visible.

    Parameters
    ----------
    df : pd.DataFrame
        A square DataFrame with same string labels for index and columns.
    figsize : tuple
        Figure size (width, height). Increase if labels overlap.
    cmap : str
        Colormap name for matplotlib.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("DataFrame must be square (same number of rows and columns).")
    if not np.all(df.index == df.columns):
        # logger.warning("Index and columns differ; continuing but axis labels may mismatch.")
        raise ValueError("Index and columns must be the same")

    df2 = df.dropna()
    if figsize is None:
        figsize = (int(0.2 * df2.shape[1]), int(0.18 * df2.shape[0]))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df2.values, cmap=cmap, aspect='auto', interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xticks(np.arange(df2.shape[1]))
    ax.set_yticks(np.arange(df2.shape[0]))
    ax.set_xticklabels(df2.columns.to_list(), rotation=90, fontsize=5)
    ax.set_yticklabels(df2.index.to_list(), fontsize=5)

    ax.set_xlabel("Columns", fontsize=8)
    ax.set_ylabel("Index", fontsize=8)
    ax.set_title(title, fontsize=10)

    plt.tight_layout(pad=0.5)
    plt.show()
