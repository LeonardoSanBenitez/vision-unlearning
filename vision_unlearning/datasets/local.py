import os
import pickle
import logging
from libs.datasets.base import UnlearnDataset, UnlearnDatasetSplit

 
class UnlearnDatasetLocal(UnlearnDataset):
    path: str

    def _load(self) -> None:
        try:
            files = [f for f in os.listdir(self.path) if f.endswith('.pkl')]
            for file in files:
                with open(os.path.join(self.path, file), 'rb') as f:
                    self._dataset_splits[UnlearnDatasetSplit(file[:-4])] = pickle.load(f)
        except Exception as e:
            logging.error(f"Error while loading the dataset: {e}")
            raise e
