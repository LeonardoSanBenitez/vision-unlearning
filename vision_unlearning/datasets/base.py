import os
import pickle
import json
from enum import Enum
from typing import Dict, List, Optional, Union, Literal
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Subset, DataLoader



class UnlearnDatasetSplit(Enum):
    Train = "train"
    Validation = "validation"
    Test = "test"
    Train_retain = "train_retain"
    Train_retain_MIA = "train_retain_mia"
    Train_forget = "train_forget"
    Test_retain = "test_retain"
    Test_forget = "test_forget"
    Validation_retain = "validation_retain"
    Validation_forget = "validation_forget"


class UnlearnDatasetSplitMode(Enum):
    Class = "class"
    Random = "random"
    Temporal = "temporal"


class SplitNotAvailableError(Exception):
    pass


class UnlearnDataset(BaseModel, ABC):
    '''
    Wrapper around huggingface datasets
    Organize the forget-retain splits
    '''
    model_config = ConfigDict(arbitrary_types_allowed=True)

    split_mode: UnlearnDatasetSplitMode
    split_kwargs: dict  # Should contain the required by the mode-specific downstream methods (_split_class, _split_random, _split_temporal)

    _dataset_splits: Dict[UnlearnDatasetSplit, Union[Subset, VisionDataset]] = {}
    _classes: Optional[List[str]] = None
    _n_classes: int = 0

    def model_post_init(self, __context: dict) -> None:
        # TODO: using pydantic's model_post_init makes this hard to debug... maybe just overwritting the constructor is better
        self._load()
        self._split()
        assert set([e.value for e in UnlearnDatasetSplit]) == set([key.value for key in list(self._dataset_splits.keys())]), "All possible splits should be filled"
        pass

    @abstractmethod
    def _load(self) -> None:
        '''
        Load the dataset from disk or download it.
        Side effects: updates the properties _dataset_splits, _classes, _n_classes
        '''
        pass

    def _split(self) -> None:
        '''
        Split the dataset based on the specified mode.
        Side effects: updates the property dataset_splits
        Raised exceptions: none
        '''
        if self.split_mode == UnlearnDatasetSplitMode.Class:
            self._split_class(**self.split_kwargs)
        elif self.split_mode == UnlearnDatasetSplitMode.Random:
            self._split_random(**self.split_kwargs)
        elif self.split_mode == UnlearnDatasetSplitMode.Temporal:
            self._split_temporal(**self.split_kwargs)
        else:
            raise NotImplementedError(f"Split mode {self.split_mode} not implemented")


    def _split_class(self, forget: List[str|int]|str|int) -> None:
        assert self._classes is not None, "Classes should be loaded before splitting"
        c: List[int]
        if isinstance(forget, list):
            if all([isinstance(e, str) for e in forget]):
                c = [i for i, class_name in enumerate(self._classes) if class_name in forget]
            elif all([isinstance(e, int) for e in forget]):
                assert type(self._n_classes) == int, "Number of classes should be loaded before splitting"
                assert all([type(e) == int for e in forget])
                max_index = np.max(forget)
                assert type(max_index) == int, "Forget should be a list of strings or integers"
                assert max_index < self._n_classes, "Forget should be a list of strings or integers"
                c = forget
            else:
                raise ValueError("Forget should be a list of strings or integers")
        elif isinstance(forget, str):
            c = [i for i, class_name in enumerate(self._classes) if class_name == forget]
        elif isinstance(forget, int):
            c = [forget] if forget < self._n_classes else None
        else:
            raise ValueError("Forget should be a list of strings or integers")

        assert type(c) == list
        assert all([type(e) == int for e in c])
        assert len(c) > 0, "Forget should be a list integers"
        
        assert isinstance(self._dataset_splits[UnlearnDatasetSplit.Train], VisionDataset), "Train should be a VisionDataset"
        assert isinstance(self._dataset_splits[UnlearnDatasetSplit.Validation], VisionDataset), "Valid should be a VisionDataset"
        assert isinstance(self._dataset_splits[UnlearnDatasetSplit.Test], VisionDataset), "Test should be a VisionDataset"
        trainf_mask = np.isin(np.array(self._dataset_splits[UnlearnDatasetSplit.Train].targets), c)  # mypy: ignore
        validf_mask = np.isin(np.array(self._dataset_splits[UnlearnDatasetSplit.Validation].targets), c)  # mypy: ignore
        testf_mask = np.isin(np.array(self._dataset_splits[UnlearnDatasetSplit.Test].targets), c)  # mypy: ignore

        train_idx = np.array(range(len(self._dataset_splits[UnlearnDatasetSplit.Train])))
        valid_idx = np.array(range(len(self._dataset_splits[UnlearnDatasetSplit.Validation])))
        test_idx = np.array(range(len(self._dataset_splits[UnlearnDatasetSplit.Test])))

        train_f_idx = train_idx[trainf_mask]
        train_r_idx = train_idx[~trainf_mask]
        valid_f_idx = valid_idx[validf_mask]
        valid_r_idx = valid_idx[~validf_mask]
        test_f_idx = test_idx[testf_mask]
        test_r_idx = test_idx[~testf_mask]

        idxs_mia = np.random.choice(train_r_idx, len(test_r_idx), replace=False)

        new_splits = {
            UnlearnDatasetSplit.Train_retain: Subset(self._dataset_splits[UnlearnDatasetSplit.Train], list(train_r_idx)),
            UnlearnDatasetSplit.Train_retain_MIA: Subset(self._dataset_splits[UnlearnDatasetSplit.Train], list(idxs_mia)),
            UnlearnDatasetSplit.Train_forget: Subset(self._dataset_splits[UnlearnDatasetSplit.Train], list(train_f_idx)),
            UnlearnDatasetSplit.Validation_retain: Subset(self._dataset_splits[UnlearnDatasetSplit.Validation], list(valid_r_idx)),
            UnlearnDatasetSplit.Validation_forget: Subset(self._dataset_splits[UnlearnDatasetSplit.Validation], list(valid_f_idx)),
            UnlearnDatasetSplit.Test_retain: Subset(self._dataset_splits[UnlearnDatasetSplit.Test], list(test_r_idx)),
            UnlearnDatasetSplit.Test_forget: Subset(self._dataset_splits[UnlearnDatasetSplit.Test], list(test_f_idx)),
        }

        self._dataset_splits.update(new_splits)

    def _split_random(self, n_forget: int, seed: int = 42) -> None:
        raise NotImplementedError("Random split not implemented")

    def _split_temporal(self, n_forget: int) -> None:
        raise NotImplementedError("Temporal split not implemented")

    def get_loader(self, split: UnlearnDatasetSplit, batchsize: int, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = True) -> Optional[DataLoader]:
        '''
        Return this split for this dataset.
        Side effects: none
        Raised exceptions: SplitNotAvailableError, if the requested split is not available
        '''
        if split not in self._dataset_splits:
            raise SplitNotAvailableError(f"Split {split} not available")
        return DataLoader(self._dataset_splits[split], batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def get_splits(self) -> Dict[UnlearnDatasetSplit, Union[Subset, VisionDataset]]:
        '''
        Return the available splits.
        Side effects: none
        Raised exceptions: none
        '''
        return self._dataset_splits

    def save(self, path: str, format: Literal['pkl', 'jpg'] = 'pkl', save_unsplit: bool = False) -> None:
        '''
        Save each split to disk.
        Side effects: saves files to disk
        Raised exceptions: OS-related errors
        '''
        assert self._classes is not None, "Classes should be loaded before saving"
        os.makedirs(path, exist_ok=True)
        for split, data in self._dataset_splits.items():
            if save_unsplit or (split != UnlearnDatasetSplit.Train and split != UnlearnDatasetSplit.Validation and split != UnlearnDatasetSplit.Test):
                if format == 'pkl':
                    with open(os.path.join(path, f"{split.value}.pkl"), 'wb') as f:
                        pickle.dump(data, f)
                elif format == 'jpg':            
                    split_path = os.path.join(path, split.value)
                    os.makedirs(split_path, exist_ok=True)
                    metadata = []
                    for idx in range(len(data)):
                        image, label = data[idx]
                        assert type(label) == int
                        image_path = os.path.join(split_path, f"{idx}.jpg")
                        # Convert tensor to PIL image and save
                        if isinstance(image, torch.Tensor):
                            image = transforms.ToPILImage()(image)
                        image.save(image_path)
                        metadata.append({
                            "file_name": f"{idx}.jpg",
                            "text": self.make_prompt_for_label(label)
                        })
                    with open(os.path.join(split_path, "metadata.jsonl"), 'w') as f:
                        for entry in metadata:
                            f.write(json.dumps(entry) + "\n")
                else:
                    raise ValueError(f"Format {format} not supported")

    def make_prompt_for_label(self, label: int) -> str:
        return f"an image of {self._classes[label]}"
