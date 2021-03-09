import numpy as np
import pandas as pd


class PriceChangeExtractor:
    name = "price_change"

    def __init__(self, dataset: np.ndarray, mid_prices: np.ndarray) -> None:
        self.dataset = dataset

        self.feature_data = pd.Series(mid_prices).pct_change()
        self.feature_data[0] = self.feature_data[1]