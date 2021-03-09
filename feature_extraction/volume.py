import numpy as np


class BidVolumeExtractor:
    name = "bid_volume"

    def __init__(self, dataset: np.ndarray) -> None:
        self.dataset = dataset

        volumes = self.dataset[:, 0, 1, :]
        volumes = volumes.sum(axis=1)

        self.feature_data = volumes


class AskVolumeExtractor:
    name = "ask_volume"

    def __init__(self, dataset: np.ndarray) -> None:
        self.dataset = dataset

        volumes = self.dataset[:, 1, 1, :]
        volumes = volumes.sum(axis=1)

        self.feature_data = volumes


class VolumeDifferenceExtractor:
    name = "volume_difference"

    def __init__(
        self,
        dataset: np.ndarray,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> None:
        self.dataset = dataset

        volume_difference = np.abs(ask_volumes - bid_volumes)

        self.feature_data = volume_difference
