from typing import ClassVar
import os
import requests
import tarfile
import numpy as np
from torchvision import datasets, transforms
from libs.datasets.base import UnlearnDataset, UnlearnDatasetSplit


class UnlearnDatasetImagenette(UnlearnDataset):
    download_path: str  # New parameter for temporary download folder

    class_mapping: ClassVar[dict] = {  # ID to human readable
        "n01440764": "tench",
        "n02102040": "english_springer",
        "n02979186": "cassette_player",
        "n03000684": "chain_saw",
        "n03028079": "church",
        "n03394916": "french_horn",
        "n03417042": "garbage_truck",
        "n03425413": "gas_pump",
        "n03445777": "golf_ball",
        "n03888257": "parachute",
    }

    def _download_imagenette(self, temp_download_path: str) -> None:
        '''
        Download Imagenette from FastAI's S3 bucket to a temporary folder.
        Skip download if files already exist.
        '''
        os.makedirs(temp_download_path, exist_ok=True)
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        tar_path = os.path.join(temp_download_path, "imagenette2-320.tgz")
        extracted_path = os.path.join(temp_download_path, "imagenette2-320")

        # Check if the dataset is already downloaded and extracted
        if os.path.exists(extracted_path) and os.path.isdir(extracted_path):
            print("Dataset already downloaded and extracted. Skipping download.")
            return

        # Download the dataset if not already downloaded
        if not os.path.exists(tar_path):
            print("Downloading Imagenette...")
            response = requests.get(url, stream=True)
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")

        # Extract the dataset
        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=temp_download_path)
        print("Extraction complete.")

    def _load(self) -> None:
        # Download Imagenette to the temporary folder
        self._download_imagenette(self.download_path)

        # Load the dataset from the extracted folder
        extracted_path = os.path.join(self.download_path, "imagenette2-320")
        train_path = os.path.join(extracted_path, "train")
        val_path = os.path.join(extracted_path, "val")

        # Define the transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Load the dataset using ImageFolder
        train_set = datasets.ImageFolder(train_path, transform=transform)
        test_set = datasets.ImageFolder(val_path, transform=transform)

        self._classes = train_set.classes
        self._n_classes = len(self._classes)

        # Split the training set into training and validation
        rng = np.random.RandomState(42)
        val_idxs = []
        for i in range(self._n_classes):
            class_idx = np.where(np.array(train_set.targets) == i)[0]
            val_idxs.append(rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False))
        val_idxs = np.hstack(val_idxs)
        train_idxs = list(set(range(len(train_set))) - set(val_idxs))

        # Create validation set as a new ImageFolder instance
        valid_data = [train_set.samples[i] for i in val_idxs]
        valid_targets = [train_set.targets[i] for i in val_idxs]

        valid_set = datasets.ImageFolder(
            train_path,
            transform=transform,
            loader=lambda x: train_set.loader(x),  # Use the same loader as the training set
        )
        valid_set.samples = valid_data
        valid_set.targets = valid_targets
        valid_set.imgs = valid_data  # Update the internal list of images

        # Update training set to exclude validation samples
        train_set.samples = [train_set.samples[i] for i in train_idxs]
        train_set.targets = [train_set.targets[i] for i in train_idxs]
        train_set.imgs = train_set.samples  # Update the internal list of images

        self._dataset_splits = {
            UnlearnDatasetSplit.Train: train_set,
            UnlearnDatasetSplit.Validation: valid_set,
            UnlearnDatasetSplit.Test: test_set,
        }

    def make_prompt_for_label(self, label: int) -> str:
        return f"an image of {self.class_mapping[self._classes[label]]}"
