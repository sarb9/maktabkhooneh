import numpy as np


class SpreadExtractor:
    name = "spread"

    def __init__(self, dataset, mid_prices) -> None:
        self.dataset = dataset

        self.feature_data = abs(dataset[:, 1, 0, 0] - dataset[:, 0, 0, 0]) / mid_prices
