'''
These datasets do not follow YET the standard Dataset interface.
They are more like a set of loosely related functions.
In any case, they may help you get started
'''
import os
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Tuple
import re
import unicodedata
import urllib.request
import bz2
from PIL import Image

import pandas as pd
import numpy as np
from unidecode import unidecode
from datasets import load_dataset

from vision_unlearning.datasets.base import logger


################################
# General utilities
################################
def create_metadata_jsonl(folder: Path):
    """
    Create metadata.jsonl for all .jpg images in `folder`.
    Each line: {"file_name": "<filename>", "text": "<class_name>"}
    """
    out_file = folder / "metadata.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for img_path in sorted(folder.glob("*.jpg")):
            name = img_path.name
            class_name = "_".join(name.split("_")[:-1])  # remove last part (e.g., 0001.jpg)
            record = {"file_name": name, "text": class_name}
            f.write(json.dumps(record) + "\n")
    print(f"Created {out_file}")


def jsonl_dump(data: list, path: str) -> None:
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def jsonl_load(path: str) -> list:
    with open(path, "r") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    return data


def balanced_subsample_lib(
    df: pd.DataFrame,
    group_cols: List[str],
    priority_col: str,  # TODO: allow for none, for random sampling
    target: int = 100,
    random_state: int = 42,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Subsample `df` without replacement to produce `target` rows (or as many as available)
    balanced as evenly as possible across the combinations of `group_cols` (i.e. strata).

    Within each stratum, the top rows are selected by highest `dataset_n_original`.
    Final output is globally ordered by decreasing `priority_col`.

    Every stratum (combiantion of group_cols) gets as close as possible to equal share. Does not give preference to any one group.
    Balances stratum-by-stratum (intersectional fairness), which may overall groups to be underpresented if there isn't enoguh data

    index is dropped (TODO refactor?)
    """
    if dropna:
        df2 = df.dropna(subset=group_cols).copy()
    else:
        df2 = df.copy()
        for c in group_cols:
            df2[c] = df2[c].fillna('NA')

    # synthetic label for each stratum
    df2['__strata__'] = df2[group_cols].astype(str).agg('|'.join, axis=1)

    counts = df2['__strata__'].value_counts().sort_index()
    total_available = counts.sum()
    n_cells = len(counts)

    if n_cells == 0:
        return df2.iloc[0:0].copy()

    if total_available <= target:
        return df2.sort_values(priority_col, ascending=False).reset_index(drop=True)

    # base allocation
    base = target // n_cells
    quotas = counts.clip(upper=base).copy()
    remaining = target - quotas.sum()

    # distribute leftovers fairly
    rng = np.random.RandomState(random_state)
    while remaining > 0:
        can_receive = quotas[quotas < counts]
        if can_receive.empty:
            quotas = counts.copy()
            break
        min_q = can_receive.min()
        candidates = list(can_receive[can_receive == min_q].index)
        rng.shuffle(candidates)
        for idx in candidates:
            if remaining <= 0:
                break
            if quotas.loc[idx] < counts.loc[idx]:
                quotas.loc[idx] += 1
                remaining -= 1

    # deterministic selection: top-k rows within each stratum
    parts = []
    for stratum, quota in quotas.items():
        if quota > 0:
            subset = (
                df2[df2['__strata__'] == stratum]
                .sort_values(priority_col, ascending=False)
                .head(quota)
            )
            parts.append(subset)

    result = (
        pd.concat(parts, axis=0)
          .sort_values(priority_col, ascending=False)  # global ordering
          .reset_index(drop=True)
    )

    return result.drop(columns='__strata__')


################################
# LFW (famous people)
################################

def count_classes_dataset_lfw():
    ds = load_dataset("bitmind/lfw", split="train")
    counts = Counter('_'.join(ex["filename"].split('_')[:-1]) for ex in ds)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts


def download_dataset_lfw(dataset_forget_name: str, dataset_retain_name: str, target: str, forget_max_img: int = 0, retain_max_img_per_class: int = 0, restrict_labels: Optional[List[str]] = None) -> Dict[str, int]:
    '''
    Downloads and already splits (TODO: separate that into two functions)
    @param forget_max_img: if >0, no more than this number of images will be saved for the forget set
    @param retain_max_img_per_class: if >0, will stratify the retain set such that no more images of one class are saved
    @param restrict_labels: if not none, save only those entities
    @return how many classes of each entity were saved
    '''
    ds = load_dataset("bitmind/lfw", split="train")
    os.makedirs(dataset_forget_name, exist_ok=True)
    os.makedirs(dataset_retain_name, exist_ok=True)

    # Save images
    class_to_number: Dict[str, int] = defaultdict(int)
    for ex in ds:
        filename = ex["filename"]
        person = "_".join(filename.split("_")[:-1])  # extract name
        img = ex["image"]

        if person == target:
            if (forget_max_img > 0) and (class_to_number[person] >= forget_max_img):
                continue
            class_to_number[person] += 1

            img.save(os.path.join(dataset_forget_name, filename), format="JPEG")
        else:
            # Enforce per-class limit if specified
            if (retain_max_img_per_class > 0) and (class_to_number[person] >= retain_max_img_per_class):
                continue
            if (restrict_labels is not None) and (person not in restrict_labels):
                continue
            class_to_number[person] += 1
            img.save(os.path.join(dataset_retain_name, filename), format="JPEG")

    # Create metadata (class as the caption)
    create_metadata_jsonl(Path(dataset_forget_name))
    create_metadata_jsonl(Path(dataset_retain_name))

    return class_to_number


################################
# Taras breeds
################################
def count_classes_dataset_taras_breeds(dataset_base_path: str) -> List[Tuple[str, int]]:
    # Count nubmer of files per folder in the dataset base path
    classes = []
    for folder in os.listdir(dataset_base_path):
        folder_path = os.path.join(dataset_base_path, folder)
        if folder == '.git':
            continue
        if os.path.isdir(folder_path):
            num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            classes.append((folder, num_files))
    sorted_counts = sorted(classes, key=lambda x: x[1], reverse=True)
    return sorted_counts


def split_dataset_taras_breeds(downloaded_folder: str, dataset_forget_name: str, dataset_retain_name: str, target: str, forget_max_img: int = 0, retain_max_img_per_class: int = 0, restrict_labels: Optional[List[str]] = None) -> Dict[str, int]:
    '''
    Given an already downloaded Taras Dog Breeds dataset at `downloaded_folder` (one folder per class), split images into forget and retain sets.
    @param forget_max_img: if >0, no more than this number of images will be saved for the forget set
    @param retain_max_img_per_class: if >0, will stratify the retain set such that no more images of one class are saved
    @param restrict_labels: if not none, save only those entities
    @return how many classes of each entity were saved
    '''
    os.makedirs(dataset_forget_name, exist_ok=True)
    os.makedirs(dataset_retain_name, exist_ok=True)

    # Save images
    class_to_number: Dict[str, int] = defaultdict(int)
    for label in os.listdir(downloaded_folder):
        folder_path = os.path.join(downloaded_folder, label)
        if label == '.git':
            continue
        if os.path.isdir(folder_path):
            for sample in os.listdir(folder_path):
                if not sample.lower().endswith(('.jpg')):
                    continue
                if label == target:
                    # Enforce forget set limit if specified
                    if (forget_max_img > 0) and (class_to_number[label] >= forget_max_img):
                        continue
                    class_to_number[label] += 1

                    src_path = os.path.join(folder_path, sample)
                    dst_path = os.path.join(dataset_forget_name, f"{label}_{uuid.uuid4().hex}.jpg")
                    os.symlink(os.path.abspath(src_path), dst_path)
                else:
                    # Enforce per-class limit if specified
                    if (retain_max_img_per_class > 0) and (class_to_number[label] >= retain_max_img_per_class):
                        continue
                    if (restrict_labels is not None) and (label not in restrict_labels):
                        continue
                    class_to_number[label] += 1

                    src_path = os.path.join(folder_path, sample)
                    dst_path = os.path.join(dataset_retain_name, f"{label}_{uuid.uuid4().hex}.jpg")
                    os.symlink(os.path.abspath(src_path), dst_path)
                # print(f"Saving the {class_to_number[label]}th image of class {label}")

    # Create metadata (class as the caption)
    create_metadata_jsonl(Path(dataset_forget_name))
    create_metadata_jsonl(Path(dataset_retain_name))

    return class_to_number


def download_dataset_taras_breeds(dataset_base_path: str, cache_folder: str) -> None:
    '''
    Do not perform splits
    Already convert images to jpg
    Do not overwrite if already exists (TODO: paremetrize?)
    '''

    if not os.path.exists(dataset_base_path):
        if not os.path.exists(cache_folder):
            subprocess.run(["git", "clone", "https://github.com/AtharvaTaras/Dog-Breeds-Dataset", cache_folder], check=True)

        # Convert all to jpg
        for root, dirs, files in os.walk(cache_folder):
            # TODO: do not transverse the root of the repo, only folders
            # Copy csv in the root in a more elegant way

            if ".git" in root:
                continue

            # figure out the relative path to recreate structure
            rel_path = os.path.relpath(root, cache_folder)
            dst_dir = os.path.join(dataset_base_path, rel_path)
            os.makedirs(dst_dir, exist_ok=True)

            for file in files:
                src_path = os.path.join(root, file)
                filename, ext = os.path.splitext(file)
                ext = ext.lower()

                # define destination path (always .jpg extension)
                dst_path = os.path.join(dst_dir, filename + ".jpg")

                try:
                    if ext in [".jpg", ".jpeg"]:
                        # just copy if already jpg
                        shutil.copy2(src_path, dst_path)
                    else:
                        # open and convert to RGB JPEG
                        with Image.open(src_path) as im:
                            im.convert("RGB").save(dst_path, "JPEG", quality=95)
                except Exception as e:
                    print(f"Skipping {src_path}: {e}")
        shutil.copy(os.path.join(cache_folder, "FCI Breeds.csv"), os.path.join(dataset_base_path, "FCI Breeds.csv"))
    else:
        logger.info('Already exists, not downloading')

    # TODO
    # return pd.read_csv(os.path.join(dataset_base_path, 'FCI Breeds.csv'), index_col='id')


################################
# AKC (attribute only)
################################

def normalize_string(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = unidecode(s)
    s = s.casefold()
    s = re.sub(r"(.)\1+", r"\1", s)   # collapse repeats
    s = re.sub(r"\s+", " ", s).strip()
    return s


def akc_find_closest_match(df, name: str, group: str) -> Optional[str]:
    assert 'group_akc' in df.columns, 'Dataframe must be AKC data'
    assert 'group_pawsome' in df.columns, 'Dataframe must have been enriched with Pawsome data'
    assert 'Other Names' in df.columns, 'Dataframe must have been enriched with Pawsome data'
    name_original = name

    # Exact match
    if (df['name_akc'] == name).sum() == 1:
        return name
    elif (df['name_akc'] == name).sum() > 1:
        logger.warning(f'Entity {name} is duplicated in df_temp, by exact match')
        return name

    # Normalized exact match
    df_temp = df.copy()
    df_temp['name_akc_normalized'] = df_temp['name_akc'].str.upper().str.replace('DOG', '').str.strip()
    name = name.upper().replace('DOG', '').strip()
    if (df_temp['name_akc_normalized'] == name).sum() == 1:
        return df_temp[df_temp['name_akc_normalized'] == name].iloc[0]['name_akc']
    elif (df_temp['name_akc_normalized'] == name).sum() > 1:
        logger.warning(f'Entity {name} is duplicated in df_temp, by normalized exact match')
        return df_temp[df_temp['name_akc_normalized'] == name].iloc[0]['name_akc']

    # Normalized exact match in synonyms
    # name_akc still contains the original correct name
    # All names appear in Other Names (one per row), including the original name
    df_temp['Other Names'] = df_temp['Other Names'].astype(str)
    df_temp['Other Names'] += ', ' + df_temp['name_akc']
    df_temp['Other Names'] = df_temp['Other Names'].str.split(',')
    df_temp = df_temp.explode('Other Names')
    df_temp = df_temp[df_temp['Other Names'] != 'nan']  # rows without synonyms were duplicated
    df_temp['Other Names'] = df_temp['Other Names'].str.upper().str.replace('DOG', '').str.strip()
    if (df_temp['Other Names'] == name).sum() == 1:
        return df_temp[df_temp['Other Names'] == name].iloc[0]['name_akc']
    elif (df_temp['Other Names'] == name).sum() > 1:
        logger.warning(f'Entity {name} is duplicated in df_temp, by normalized exact match')
        return df_temp[df_temp['Other Names'] == name].iloc[0]['name_akc']

    # Aggressive matching
    name = normalize_string(name)
    df_temp['Other Names'] = df_temp['Other Names'].apply(normalize_string)
    for subname in name.split('('):
        subname = subname.replace('(', '').replace(')', '').strip()
        if (df_temp['Other Names'] == name).sum() == 1:
            return df_temp[df_temp['Other Names'] == name].iloc[0]['name_akc']
        elif (df_temp['Other Names'] == name).sum() > 1:
            logger.warning(f'Entity {name} is duplicated in df_temp, by normalized exact match')
            return df_temp[df_temp['Other Names'] == name].iloc[0]['name_akc']

    # OR matching
    # Check how many words each df_temp['Other Names'] contains from name
    # The entire subword doesn't have to match, just be contained
    # For example, 'hungar point viz' counts as having two matches with 'hungarian pointing vizsla'
    # If one option is isolated the best match (the only one with n matches), return it
    df_temp['Other Names'] = df_temp['Other Names'] + ' ' + df_temp['group_akc'].apply(normalize_string)
    name += ' ' + normalize_string(group.replace('sheepdogs', 'shep').replace('catledogs', 'catle').replace('pointing', 'point').replace('terriers', 'terrier').replace('retrievers', 'retriever').replace('sighthounds', 'sight').replace('dachshunds', 'dachs').replace('_and', '').replace('_related_breeds', '').replace('_primitive_types', ''))  # noqa
    name = name.replace('(', '').replace(')', '').replace('_', ' ').replace('-', ' ')
    name = name.replace('ian ', ' ').replace('ish ', ' ').replace('ese ', ' ').replace('an ', ' ').replace('en ', ' ').replace('er ', ' ').replace('s ', ' ')
    name_words = set(name.split())
    df_temp['match_count'] = df_temp['Other Names'].apply(lambda x: sum(word in x for word in name_words))

    for n in range(2, 10):
        filtered = df_temp[df_temp['match_count'] >= n]
        if len(filtered) == 1:
            return filtered.iloc[0]['name_akc']

    # print(name_original)
    # print(filtered[['name_akc', 'Other Names', 'match_count']])
    logger.error(f"Could not match breed {name_original}")
    return None


def download_dataset_akc(output_path: str) -> pd.DataFrame:
    # TODO: make output_path optional, and save to temp file?
    url = "https://raw.githubusercontent.com/tmfilho/akcdata/refs/heads/master/data/akc-data-latest.csv"

    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
    return pd.read_csv(output_path)


################################
# Pantheon (attribute only)
################################

def download_dataset_pantheon(
    url: str = "https://storage.googleapis.com/pantheon-public-data/person_2025_update.csv.bz2",
    bz2_path: str = "assets/datasets/person_2025_update.csv.bz2",
    csv_path: str = "assets/datasets/person_2025_update.csv",
) -> pd.DataFrame:
    # https://pantheon.world/data/datasets
    if not os.path.exists(bz2_path):
        urllib.request.urlretrieve(url, bz2_path)
        with bz2.open(bz2_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                f_out.write(f_in.read())
        logger.info(f'Downloaded {csv_path}')
    return pd.read_csv(csv_path)


def pantheon_find_closest_match(df, name: str) -> Optional[str]:
    if (df['slug'] == name).sum() == 1:
        return name
    elif (df['slug'] == name).sum() > 1:
        logger.warning(f'Entity {name} is duplicated in pantheon, by exact match')
        return name

    df_temp = df.copy()  # TODO mistake?
    for subname in name.split('_'):
        df_temp = df_temp[df_temp['name'].str.contains(subname)]
    if df_temp.shape[0] == 1:
        return df_temp['slug'].iloc[0]

    df_temp = df_temp[df_temp['name'].str.count(' ') == name.count('_')]
    if df_temp.shape[0] == 1:
        return df_temp['slug'].iloc[0]
    if df_temp.shape[0] > 1:
        logger.warning(f'Entity {name} is duplicated in pantheon, by AND match')
        print(df_temp)
        return df_temp['slug'].iloc[0]

    # TODO: there are more things that can be done, like substituting oe by ö
    logger.warning(f'Entity {name} not in pantheon')
    return None
