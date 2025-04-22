from typing import List
import copy
import numpy as np
from torchvision import datasets, transforms
from vision_unlearning.datasets.base import UnlearnDataset, UnlearnDatasetSplit


class UnlearnDatasetCifar(UnlearnDataset):
    download_path: str

    def _load(self) -> None:
        # Define the transform
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        train_set = datasets.CIFAR10(self.download_path, train=True, transform=transform, download=True)
        test_set = datasets.CIFAR10(self.download_path, train=False, transform=transform, download=True)

        self._classes = train_set.classes
        assert self._classes is not None
        self._n_classes = len(self._classes)

        rng = np.random.RandomState(42)
        val_idxs = []
        for i in range(self._n_classes):
            class_idx = np.where(np.array(train_set.targets) == i)[0]
            val_idxs.append(rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False))
        val_idxs_stacked = np.hstack(val_idxs)
        train_idxs = list(set(range(len(train_set))) - set(val_idxs_stacked))

        valid = copy.deepcopy(train_set)
        train = copy.deepcopy(train_set)

        valid.data = train_set.data[val_idxs_stacked]
        valid.targets = list(np.array(train_set.targets)[val_idxs_stacked])

        train.data = train_set.data[train_idxs]
        train.targets = list(np.array(train_set.targets)[train_idxs])

        self._dataset_splits = {
            UnlearnDatasetSplit.Train: train,
            UnlearnDatasetSplit.Validation: valid,
            UnlearnDatasetSplit.Test: test_set,
        }
