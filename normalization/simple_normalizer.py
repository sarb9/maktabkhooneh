import numpy as np


class SimpleOrderBookNormalizer:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def simple_normalize(self) -> np.ndarray:

        bid_mean = np.mean(self.data[:, 0, 1, :], axis=(0))
        ask_mean = np.mean(self.data[:, 1, 1, :], axis=(0))

        bid_var = np.var(self.data[:, 0, 1, :], axis=(0))
        ask_var = np.var(self.data[:, 1, 1, :], axis=(0))

        data = np.copy(self.data)

        data[:, 0, 1, :] = (data[:, 0, 1, :] - bid_mean) / bid_var
        data[:, 1, 1, :] = (data[:, 1, 1, :] - ask_mean) / ask_var

        return data
