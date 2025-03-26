from libs.datasets.base import UnlearnDataset


class UnlearnDatasetCoco(UnlearnDataset):
    def _load(self) -> None:
        raise NotImplementedError("COCO dataset loading not implemented")
