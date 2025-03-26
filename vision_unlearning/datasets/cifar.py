from typing import List
import copy
import numpy as np
from torchvision import datasets, transforms
from libs.datasets.base import UnlearnDataset, UnlearnDatasetSplit


class UnlearnDatasetCifar(UnlearnDataset):
    download_path: str

    def _load(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_set = datasets.CIFAR10(self.download_path, train=True, transform=transform, download=True)
        test_set = datasets.CIFAR10(self.download_path, train=False, transform=transform, download=True)

        self._classes = train_set.classes
        self._n_classes = len(self._classes)

        rng = np.random.RandomState(42)
        val_idxs = []
        for i in range(self._n_classes):
            class_idx = np.where(np.array(train_set.targets) == i)[0]
            val_idxs.append(rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False))
        val_idxs = np.hstack(val_idxs)
        train_idxs = list(set(range(len(train_set))) - set(val_idxs))

        valid = copy.deepcopy(train_set)
        train = copy.deepcopy(train_set)

        valid.data = train_set.data[val_idxs]
        valid.targets = list(np.array(train_set.targets)[val_idxs])

        train.data = train_set.data[train_idxs]
        train.targets = list(np.array(train_set.targets)[train_idxs])

        self._dataset_splits = {
            UnlearnDatasetSplit.Train: train,
            UnlearnDatasetSplit.Validation: valid,
            UnlearnDatasetSplit.Test: test_set,
        }
